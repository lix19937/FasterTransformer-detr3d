/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "src/fastertransformer/layers/attention_layers/SVCrossAttentionLayer.h"
#include "src/fastertransformer/kernels/svcross_attention_kernels.h"
#include "src/fastertransformer/models/sv/helper_macros.h"

namespace fastertransformer {

template<typename T>
void SVCrossAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                       const std::vector<fastertransformer::Tensor>* input_tensors,
                                       const CAttentionWeight<T>* weights,
                                       cudaStream_t stream)
{
    stream_ = stream;
    __forward(output_tensors, input_tensors, weights);
    sync_check_cuda_error();
}

template<typename T>
SVCrossAttentionLayer<T>::SVCrossAttentionLayer(size_t max_batch_size,
                                                size_t max_seq_len,
                                                size_t hidden_units,
                                                size_t num_cam,
                                                size_t l2i_h,
                                                size_t l2i_w,
                                                int* img_shape,
                                                float* pc_range,
                                                float q_scaling,
                                                cudaStream_t stream,
                                                cublasMMWrapper* cublas_wrapper,
                                                IAllocator* allocator,
                                                bool is_free_buffer_after_forward,
                                                bool sparse,
                                                int int8_mode):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    hidden_units_(hidden_units),
    num_cam_(num_cam),
    l2i_h_(l2i_h),
    l2i_w_(l2i_w),
    sparse_(sparse),
    q_scaling_(q_scaling),
    int8_mode_(int8_mode)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    ::memcpy(img_shape_, img_shape, 2 * sizeof(int));
    ::memcpy(pc_range_, pc_range, 6 * sizeof(float));
    num_query_ = max_seq_len_;
    allocateBuffer();
    check_cuda_error(cudaStreamCreate(&inner_stream_));
    check_cuda_error(cudaEventCreate(&inner_event_));
    check_cuda_error(cudaEventCreate(&event_base_));
}

template<typename T>
SVCrossAttentionLayer<T>::SVCrossAttentionLayer(SVCrossAttentionLayer<T> const& layer):
    BaseLayer(layer.stream_,
              layer.cublas_wrapper_,
              layer.allocator_,
              layer.is_free_buffer_after_forward_,
              layer.cuda_device_prop_,
              layer.sparse_),
    max_batch_size_(layer.max_batch_size_),
    max_seq_len_(layer.max_seq_len_),
    hidden_units_(layer.hidden_units_),
    sparse_(layer.sparse_),
    q_scaling_(layer.q_scaling_),
    int8_mode_(layer.int8_mode_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    check_cuda_error(cudaStreamCreate(&inner_stream_));
    check_cuda_error(cudaEventCreate(&inner_event_));
    check_cuda_error(cudaEventCreate(&event_base_));
}

template<typename T>
SVCrossAttentionLayer<T>::~SVCrossAttentionLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
    check_cuda_error(cudaStreamDestroy(inner_stream_));
    check_cuda_error(cudaEventDestroy(inner_event_));
    check_cuda_error(cudaEventDestroy(event_base_));
}

template<typename T>
void SVCrossAttentionLayer<T>::allocateBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    d_img_shape_ = (int*)allocator_->malloc(sizeof(int) * 2, false);
    check_cuda_error(cudaMemcpy(d_img_shape_, img_shape_, sizeof(int) * 2, cudaMemcpyHostToDevice));

    d_pc_range_ = (float*)allocator_->malloc(sizeof(float) * 6, false);
    check_cuda_error(cudaMemcpy(d_pc_range_, pc_range_, 6 * sizeof(float), cudaMemcpyHostToDevice));

    if (std::is_same<T, float>::value) {
        d_rp_norm_ = (T*)allocator_->malloc(sizeof(T) * 1 * num_query_ * 4, false);
        d_rp_matmuled_ = (T*)allocator_->malloc(sizeof(T) * 1 * num_cam_ * 4 * num_query_, false);
        d_rpc_norm_ = (T*)allocator_->malloc(sizeof(T) * 1 * num_cam_ * num_query_ * 2, false);
        d_l2i_norm_ = (T*)allocator_->malloc(sizeof(T) * num_cam_ * l2i_h_ * l2i_w_, false);
        d_fs_output_ = (T*)allocator_->malloc(sizeof(T) * ch_ * num_query_ * num_cam_ * 1 * 4, false);
        d_fs_mask_ = (uint8_t*)allocator_->malloc(sizeof(uint8_t) * ch_ * num_query_ * num_cam_ * 1 * 1, false);
        d_attention_weights_output_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * num_cam_ * 1 * 4, false);
        d_reduce_output_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * ch_, false);

        d_pencoder_bufs_0_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * 3, false);
        d_pencoder_bufs_1_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * hidden_units_, false);
    }
    else if (std::is_same<T, half>::value) {
        d_rp_norm_ = (T*)allocator_->malloc(sizeof(T) * 1 * num_query_ * 4, false);
        d_rp_matmuled_ = (T*)allocator_->malloc(sizeof(T) * 1 * num_cam_ * 4 * num_query_, false);

        d_rpc_norm_ = (T*)allocator_->malloc(sizeof(T) * 1 * num_cam_ * num_query_ * 2, false);
        d_l2i_norm_ = (T*)allocator_->malloc(sizeof(T) * num_cam_ * l2i_h_ * l2i_w_, false);
        d_fs_output_ = (T*)allocator_->malloc(sizeof(T) * ch_ * num_query_ * num_cam_ * 1 * 4, false);
        d_fs_mask_ = (uint8_t*)allocator_->malloc(sizeof(uint8_t) * ch_ * num_query_ * num_cam_ * 1 * 1, false);
        d_attention_weights_output_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * num_cam_ * 1 * 4, false);
        d_reduce_output_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * ch_, false);

        d_pencoder_bufs_0_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * 3, false);
        d_pencoder_bufs_1_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * hidden_units_, false);
    }

    // pos encoder
    d_pencoder_bufs_[0] = d_pencoder_bufs_0_;
    d_pencoder_bufs_[1] = d_pencoder_bufs_1_;
}

template<typename T>
void SVCrossAttentionLayer<T>::freeBuffer()
{
    allocator_->free(d_img_shape_);
    allocator_->free(d_pc_range_);
    allocator_->free(d_rp_norm_);
    allocator_->free(d_rp_matmuled_);
    allocator_->free(d_rpc_norm_);
    allocator_->free(d_l2i_norm_);
    allocator_->free(d_fs_output_);
    allocator_->free(d_fs_mask_);
    allocator_->free(d_attention_weights_output_);
    allocator_->free(d_reduce_output_);

    allocator_->free(d_pencoder_bufs_0_);
    allocator_->free(d_pencoder_bufs_1_);
}

template class SVCrossAttentionLayer<float>;
template class SVCrossAttentionLayer<half>;

}  // namespace fastertransformer
