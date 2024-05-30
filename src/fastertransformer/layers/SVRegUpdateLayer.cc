/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "src/fastertransformer/layers/SVRegUpdateLayer.h"
#include "src/fastertransformer/kernels/svreg_update_kernels.h"

namespace fastertransformer {

template<typename T>
void SVRegUpdateLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                  const std::vector<fastertransformer::Tensor>* input_tensors,
                                  const RegBranchWeight<T>* weights,
                                  cudaStream_t stream)
{
    stream_ = stream;
    __forward(output_tensors, input_tensors, weights);
    sync_check_cuda_error();
}

template<typename T>
SVRegUpdateLayer<T>::SVRegUpdateLayer(size_t max_batch_size,
                                      size_t max_seq_len,
                                      size_t hidden_units,
                                      float *pc_range,
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
    sparse_(sparse),
    q_scaling_(q_scaling),
    int8_mode_(int8_mode)
{
    num_query_ = max_seq_len_;
    ::memcpy(pc_range_, pc_range, 6 * sizeof(float));

    allocateBuffer();
}

template<typename T>
SVRegUpdateLayer<T>::SVRegUpdateLayer(SVRegUpdateLayer<T> const& layer):
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
}

template<typename T>
SVRegUpdateLayer<T>::~SVRegUpdateLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void SVRegUpdateLayer<T>::allocateBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    fc1_buf_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * hidden_units_, false);
    fc2_buf_ = (T*)allocator_->malloc(sizeof(T) * num_query_ * hidden_units_, false);

    fc_buf_[0] = fc1_buf_;
    fc_buf_[1] = fc2_buf_;
    fc_buf_[2] = fc1_buf_;

    d_pc_range_ = (float*)allocator_->malloc(sizeof(float) * 6, false);
    check_cuda_error(cudaMemcpy(d_pc_range_, pc_range_, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

template<typename T>
void SVRegUpdateLayer<T>::freeBuffer()
{
    allocator_->free(fc1_buf_);
    allocator_->free(fc2_buf_);
    allocator_->free(d_pc_range_);
}

template class SVRegUpdateLayer<float>;
template class SVRegUpdateLayer<half>;

}  // namespace fastertransformer
