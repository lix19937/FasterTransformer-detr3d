/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "src/fastertransformer/layers/attention_layers/SVUnfusedAttentionLayer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"

namespace fastertransformer {

// SVUnfusedAttentionLayer mean mha 
template<typename T>
void SVUnfusedAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                         const std::vector<fastertransformer::Tensor>* input_tensors,
                                         const AttentionWeight<T>* attention_weights,
                                         cudaStream_t stream)
{
    // input_tensors:
    //      input_query (token_num, d_model), d_model see https://arxiv.org/pdf/1706.03762.pdf  E_q
    //      1,197(token_num),768(E_q) attention_mask (batch, 1, seqlen, seqlen), padding_offset (token_num) =patch +
    //      1 relative_attention_bias (optional)
    // If padding_offset.data is nullptr, then not remove padding
    stream_ = stream;
    
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // FT_CHECK(isValidBatchSize(input_tensors->at(1).shape[0]));
    // FT_CHECK(isValidSeqLen(input_tensors->at(1).shape[2]));
    // FT_CHECK(input_tensors->size() == 3 || input_tensors->size() == 4); 

    const int request_batch_size = input_tensors->at(1).shape[0];
    const int request_seq_len = input_tensors->at(1).shape[2];

    T* attention_out = (T*)output_tensors->at(0).data;
    const T* from_tensor = (const T*)input_tensors->at(0).data;

    const T* attention_mask = (const T*)input_tensors->at(1).data;
    const int* padding_offset = (const int*)input_tensors->at(2).data;
    const T* relative_attention_bias = input_tensors->size() == 4 ? (const T*)input_tensors->at(3).data : nullptr;

    bool use_relative_position_bias = relative_attention_bias != nullptr;

    const int m = input_tensors->at(0).shape[0];  // seqlen
    int k = d_model_;                             // eq:  hidden_units_
    int n = hidden_units_;                        
#ifdef SPARSITY_ENABLED
    int m_tmp = m;
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    const int m_padded = m_tmp;

    if (sparse_ && cublas_wrapper_->isUseSparse(1, n, m, k)) {
        cublas_wrapper_->SpGemm(
            CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, attention_weights->query_weight.sp_kernel, from_tensor, q_buf_);
        cublas_wrapper_->SpGemm(
            CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, attention_weights->key_weight.sp_kernel, from_tensor, k_buf_);
        cublas_wrapper_->SpGemm(
            CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, attention_weights->value_weight.sp_kernel, from_tensor, v_buf_);
    }
    else {
#endif
        const bool is_batched_QKV_ = cublas_wrapper_->isFuseBatchGemm(3, n, m, k);
        // printf("is_batched_QKV_:%d , use_relative_position_bias:%d\n", is_batched_QKV_? 1: 0,
        // use_relative_position_bias?1:0); //// 0
        if (is_batched_QKV_) {
            const T* hA[]{attention_weights->query_weight.kernel,
                          attention_weights->key_weight.kernel,
                          attention_weights->value_weight.kernel,
                          nullptr,
                          from_tensor,  // Q
                          from_tensor,  // K
                          from_tensor,  // V
                          nullptr,
                          q_buf_,
                          k_buf_,
                          v_buf_};
            // Note: Here, we assume the weights of each time may be different.
            // If we can preprocess these weights before inference, we can reduce the overhead
            // caused by cudaMemcpyAsync
            cudaMemcpyAsync((void*)batch_qkv_kernel_ptr_, hA, sizeof(T*) * 12, cudaMemcpyHostToDevice, stream_);
            cublas_wrapper_->batchedGemm(CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         n,
                                         m,
                                         k,
                                         (const void* const*)batch_qkv_kernel_ptr_,
                                         n,
                                         (const void* const*)batch_qkv_input_ptr_,
                                         k,
                                         (void* const*)batch_qkv_buf_ptr_,
                                         n,
                                         3);
        }
        else {  /// q_buf_ = from_tensor * query_weight.kernel
                /// enter into here
            // printf("##0 \n");  //// enter in this branch
            // in svod v no pos_query，k and q add with the pos_query
            // fusion other stage`s op
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  attention_weights->query_weight.kernel,
                                  n,
                                  from_tensor + m * k,  //// [svod]   q + query_pos --> q
                                  k,
                                  q_buf_,
                                  n);

            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  attention_weights->key_weight.kernel,
                                  n,
                                  from_tensor + m * k,  //// [svod] k + query_pos --> k
                                  k,
                                  k_buf_,
                                  n);

            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  attention_weights->value_weight.kernel,
                                  n,
                                  from_tensor,
                                  k,
                                  v_buf_,
                                  n);
        }
#ifdef SPARSITY_ENABLED
    }
#endif

    if (padding_offset == nullptr) {
        // printf("##1 \n");  //// enter in this branch  map to `Add`  
        invokeAddQKVBiasTranspose(q_buf_2_,
                                  k_buf_2_,
                                  v_buf_2_,
                                  q_buf_,
                                  attention_weights->query_weight.bias,/////
                                  k_buf_,
                                  attention_weights->key_weight.bias,
                                  v_buf_,
                                  attention_weights->value_weight.bias,
                                  request_batch_size,
                                  request_seq_len,
                                  head_num_,
                                  size_per_head_,
                                  stream_);
        sync_check_cuda_error();
    }
    else {
        printf("##2 \n");
        cudaMemsetAsync(q_buf_2_, 0, 3 * request_batch_size * request_seq_len * hidden_units_ * sizeof(T), stream_);
        sync_check_cuda_error();
        invokeAddQKVBiasRebuildPadding(q_buf_,
                                       attention_weights->query_weight.bias,
                                       k_buf_,
                                       attention_weights->key_weight.bias,
                                       v_buf_,
                                       attention_weights->value_weight.bias,
                                       q_buf_2_,
                                       k_buf_2_,
                                       v_buf_2_,
                                       request_batch_size,
                                       request_seq_len,
                                       head_num_,
                                       size_per_head_,
                                       m,
                                       padding_offset,
                                       stream_);
        sync_check_cuda_error();
    }

    float scalar = 1 / (sqrtf(size_per_head_ * 1.0f) * q_scaling_);
    // printf("##3 \n");  //// enter in this branch   map to `MatMul`
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,// transpose
                                        CUBLAS_OP_N,
                                        request_seq_len,// m
                                        request_seq_len,// n
                                        size_per_head_,// k
                                        k_buf_2_,                           /// a
                                        size_per_head_,                     /// lda
                                        request_seq_len * size_per_head_,   /// strideA
                                        q_buf_2_,                           /// b
                                        size_per_head_,                     /// ldb
                                        request_seq_len * size_per_head_,   /// strideB
                                        qk_buf_,                            /// c
                                        request_seq_len,                    /// ldc
                                        request_seq_len * request_seq_len,  /// strideC
                                        request_batch_size * head_num_,     /// batch_count
                                        scalar);//f_alpha

    // TODO (fuse with softMax lix19937)
    if (use_relative_position_bias) {  // not enter
        printf("##4 \n");
        invokeAddRelativeAttentionBias(
            qk_buf_, relative_attention_bias, request_batch_size, head_num_, request_seq_len, stream_);
    }

    invokeMaskedSoftMax(
        qk_buf_, qk_buf_, attention_mask, request_batch_size, request_seq_len, head_num_, (T)1.0f, stream_);
    sync_check_cuda_error();

    // printf("##5 \n");  //// enter in this branch
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head_,
                                        request_seq_len,
                                        request_seq_len,
                                        v_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_,
                                        request_seq_len,                    /// ldb
                                        request_seq_len * request_seq_len,  /// strideB
                                        qkv_buf_,                           /// c
                                        size_per_head_,                     // ldc
                                        request_seq_len * size_per_head_,   /// strideC
                                        request_batch_size * head_num_);    /// batch_count

    if (padding_offset == nullptr) {
        // printf("##6 \n");  //// enter in this branch
        invokeTransposeQKV(
            qkv_buf_2_, qkv_buf_, request_batch_size, request_seq_len, head_num_, size_per_head_, stream_);
        sync_check_cuda_error();
    }
    else {
        invokeTransposeAttentionOutRemovePadding(qkv_buf_,
                                                 qkv_buf_2_,
                                                 m,
                                                 request_batch_size,
                                                 request_seq_len,
                                                 head_num_,
                                                 size_per_head_,
                                                 padding_offset,
                                                 stream_);
    }

    k = hidden_units_;
    n = d_model_;

#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, n, m, k)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                n,
                                m_padded,
                                k,
                                attention_weights->attention_output_weight.sp_kernel,
                                qkv_buf_2_,
                                attention_out);
    }
    else {
#endif
        // printf("##7 \n");   //// enter in this branch
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              attention_weights->attention_output_weight.kernel,
                              n,
                              qkv_buf_2_,
                              k,
                              attention_out,
                              n);
#ifdef SPARSITY_ENABLED
    }
#endif

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
SVUnfusedAttentionLayer<T>::SVUnfusedAttentionLayer(size_t max_batch_size,
                                                    size_t max_seq_len,
                                                    size_t head_num,
                                                    size_t size_per_head,
                                                    float q_scaling,
                                                    cudaStream_t stream,
                                                    cublasMMWrapper* cublas_wrapper,
                                                    IAllocator* allocator,
                                                    bool is_free_buffer_after_forward,
                                                    bool sparse):
    SVUnfusedAttentionLayer(max_batch_size,
                            max_seq_len,
                            head_num,
                            size_per_head,
                            head_num * size_per_head,
                            q_scaling,
                            stream,
                            cublas_wrapper,
                            allocator,
                            is_free_buffer_after_forward,
                            sparse)
{
}

template<typename T>
SVUnfusedAttentionLayer<T>::SVUnfusedAttentionLayer(size_t max_batch_size,
                                                    size_t max_seq_len,
                                                    size_t head_num,
                                                    size_t size_per_head,
                                                    size_t d_model,
                                                    float q_scaling,
                                                    cudaStream_t stream,
                                                    cublasMMWrapper* cublas_wrapper,
                                                    IAllocator* allocator,
                                                    bool is_free_buffer_after_forward,
                                                    bool sparse):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    hidden_units_(head_num_ * size_per_head_),
    sparse_(sparse),
    q_scaling_(q_scaling)
{
    allocateBuffer();
}

template<typename T>
SVUnfusedAttentionLayer<T>::SVUnfusedAttentionLayer(SVUnfusedAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_),
    max_batch_size_(attention_layer.max_batch_size_),
    max_seq_len_(attention_layer.max_seq_len_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    d_model_(attention_layer.d_model_),
    hidden_units_(attention_layer.hidden_units_),
    sparse_(attention_layer.sparse_),
    q_scaling_(attention_layer.q_scaling_)
{
}

template<typename T>
SVUnfusedAttentionLayer<T>::~SVUnfusedAttentionLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void SVUnfusedAttentionLayer<T>::allocateBuffer()
{
    if (!is_allocate_buffer_) {
        q_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        k_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        v_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        q_buf_2_ = (T*)allocator_->malloc(sizeof(T) * 3 * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        k_buf_2_ = q_buf_2_ + max_batch_size_ * max_seq_len_ * hidden_units_;
        v_buf_2_ = k_buf_2_ + max_batch_size_ * max_seq_len_ * hidden_units_;
        qk_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_, false);
        qkv_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        qkv_buf_2_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        batch_qkv_kernel_ptr_ = (T**)allocator_->malloc(sizeof(T*) * 12, false);
        batch_qkv_input_ptr_ = batch_qkv_kernel_ptr_ + 4;
        batch_qkv_buf_ptr_ = batch_qkv_input_ptr_ + 4;
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void SVUnfusedAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    q_buf_ = (T*)allocator_->reMalloc(q_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    k_buf_ = (T*)allocator_->reMalloc(k_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    v_buf_ = (T*)allocator_->reMalloc(v_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * 3 * batch_size * seq_len * hidden_units_, false);
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * hidden_units_;
    qk_buf_ = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * head_num_ * seq_len * seq_len, false);
    qkv_buf_ = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    batch_qkv_kernel_ptr_ = (T**)allocator_->reMalloc(batch_qkv_kernel_ptr_, sizeof(T*) * 12, false);
    batch_qkv_input_ptr_ = batch_qkv_kernel_ptr_ + 4;
    batch_qkv_buf_ptr_ = batch_qkv_input_ptr_ + 4;
    is_allocate_buffer_ = true;
}

template<typename T>
void SVUnfusedAttentionLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free(q_buf_);
        allocator_->free(k_buf_);
        allocator_->free(v_buf_);
        allocator_->free(q_buf_2_);
        allocator_->free(qk_buf_);
        allocator_->free(qkv_buf_);
        allocator_->free(qkv_buf_2_);
        allocator_->free(batch_qkv_kernel_ptr_);
        sync_check_cuda_error();
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool SVUnfusedAttentionLayer<T>::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ < batch_size) {
        max_batch_size_ = batch_size;
    }
    return true;
}

template<typename T>
bool SVUnfusedAttentionLayer<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ < seq_len) {
        max_seq_len_ = seq_len;
    }
    return true;
}

template class SVUnfusedAttentionLayer<float>;
template class SVUnfusedAttentionLayer<half>;

}  // namespace fastertransformer