/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#pragma once

#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <vector>

namespace fastertransformer {

template<typename T>
class SVClsLayer: public BaseLayer {

    size_t max_batch_size_ = 1;
    size_t max_seq_len_ = 1;

    size_t hidden_units_;

    bool sparse_;
    float q_scaling_;
    int int8_mode_ = 0;

    size_t num_query_;

    void allocateBuffer() override;
    void freeBuffer() override;

    void __forward(std::vector<fastertransformer::Tensor>* output_tensors,
                   const std::vector<fastertransformer::Tensor>* input_tensors,
                   const ClsBranchWeight<T>* weights);

protected:
    using BaseLayer::stream_;
    using BaseLayer::is_free_buffer_after_forward_;
    using BaseLayer::is_allocate_buffer_;
    using BaseLayer::cublas_wrapper_;
    using BaseLayer::allocator_;

    T* fc1_buf_ = nullptr;  //   [L, E_d]
    T* fc2_buf_ = nullptr;  //   [L, E_d]

    T* fc_buf_[2];

public:
    SVClsLayer(size_t max_batch_size,
                     size_t max_seq_len,
                     size_t hidden_units,
                     float q_scaling,
                     cudaStream_t stream,
                     cublasMMWrapper* cublas_wrapper,
                     IAllocator* allocator,
                     bool is_free_buffer_after_forward,
                     bool sparse = false,
                     int int8_mode = 0);

    SVClsLayer(SVClsLayer<T> const& reg_update_layer);

    ~SVClsLayer();

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const ClsBranchWeight<T>* weights,
                 cudaStream_t stream);
};

}  // namespace fastertransformer
