/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#pragma once

#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"

namespace fastertransformer {

template<typename T>
class SVUnfusedAttentionLayer: public BaseAttentionLayer<T> {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;

    // metadata
    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;

    size_t hidden_units_;
    bool sparse_;
    float q_scaling_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);
    void allocateBuffer(size_t batch_size, size_t seq_len);

protected:
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

    T* q_buf_ = nullptr;
    T* k_buf_ = nullptr;
    T* v_buf_ = nullptr;
    T* q_buf_2_ = nullptr;
    T* k_buf_2_ = nullptr;
    T* v_buf_2_ = nullptr;
    T* qk_buf_ = nullptr;
    T* qkv_buf_ = nullptr;
    T* qkv_buf_2_ = nullptr;

    T** batch_qkv_kernel_ptr_ = nullptr;
    T** batch_qkv_input_ptr_ = nullptr;
    T** batch_qkv_buf_ptr_ = nullptr;

public:
    SVUnfusedAttentionLayer(size_t max_batch_size,
                            size_t max_seq_len,
                            size_t head_num,
                            size_t size_per_head,
                            float q_scaling,
                            cudaStream_t stream,
                            cublasMMWrapper* cublas_wrapper,
                            IAllocator* allocator,
                            bool is_free_buffer_after_forward,
                            bool sparse = false);

    SVUnfusedAttentionLayer(size_t max_batch_size,
                            size_t max_seq_len,
                            size_t head_num,
                            size_t size_per_head,
                            size_t d_model,
                            float q_scaling,
                            cudaStream_t stream,
                            cublasMMWrapper* cublas_wrapper,
                            IAllocator* allocator,
                            bool is_free_buffer_after_forward,
                            bool sparse = false);

    SVUnfusedAttentionLayer(SVUnfusedAttentionLayer<T> const& attention_layer);

    ~SVUnfusedAttentionLayer();

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const AttentionWeight<T>* attention_weights) override
    {
    }

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const AttentionWeight<T>* attention_weights,
                 cudaStream_t stream);
};

}  // namespace fastertransformer
