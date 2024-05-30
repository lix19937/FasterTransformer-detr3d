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
#include "src/fastertransformer/models/sv/SVWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <vector>

namespace fastertransformer {

template<typename T>
class SVCrossAttentionLayer: public BaseLayer {
    size_t max_batch_size_ = 1;
    size_t max_seq_len_ = 1;

    size_t hidden_units_;
    size_t num_cam_;
    size_t l2i_h_;
    size_t l2i_w_;
    int img_shape_[2];
    float pc_range_[6];

    bool sparse_;
    float q_scaling_;

    int int8_mode_ = 0;
    size_t num_query_;
    size_t ch_ = 256;

    cudaEvent_t inner_event_;
    cudaEvent_t event_base_;
    cudaStream_t inner_stream_;

    void allocateBuffer() override;
    void freeBuffer() override;

    void __forward(std::vector<fastertransformer::Tensor>* output_tensors,
                   const std::vector<fastertransformer::Tensor>* input_tensors,
                   const CAttentionWeight<T>* weights);

    /// fp32 sv
    void __forward_fp32_SV_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                  const std::vector<fastertransformer::Tensor>* input_tensors,
                                  const CAttentionWeight<T>* weights);
    void __forward_magic_fp32_SV_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                        const std::vector<fastertransformer::Tensor>* input_tensors,
                                        const CAttentionWeight<T>* weights,
                                        const HelperIRPara<T>* helper_weights,
                                        cudaStream_t stream);
    /// fp32 av
    void __forward_fp32_AV_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                  const std::vector<fastertransformer::Tensor>* input_tensors,
                                  const CAttentionWeight<T>* weights);
    void __forward_magic_fp32_AV_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                        const std::vector<fastertransformer::Tensor>* input_tensors,
                                        const CAttentionWeight<T>* weights,
                                        const HelperIRPara<T>* helper_weights,
                                        cudaStream_t stream);


    /// int8, int8, int8 v2
    void __forward_fake_int8_v2_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                       const std::vector<fastertransformer::Tensor>* input_tensors,
                                       const CAttentionWeight<T>* weights);
    void __forward_magic_fake_int8_v2_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                             const std::vector<fastertransformer::Tensor>* input_tensors,
                                             const CAttentionWeight<T>* weights,
                                             const HelperIRPara<T>* helper_weights,
                                             cudaStream_t stream);

    /// fp16, fp16, fp16  c/32hw32
    void __forward_half_chw32_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                     const std::vector<fastertransformer::Tensor>* input_tensors,
                                     const CAttentionWeight<T>* weights);
    void __forward_magic_half_chw32_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                           const std::vector<fastertransformer::Tensor>* input_tensors,
                                           const CAttentionWeight<T>* weights,
                                           const HelperIRPara<T>* helper_weights,
                                           cudaStream_t stream);
    /// fp16, fp16, fp16  chw
    void __forward_half_linear_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                      const std::vector<fastertransformer::Tensor>* input_tensors,
                                      const CAttentionWeight<T>* weights);
    void __forward_magic_half_linear_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                            const std::vector<fastertransformer::Tensor>* input_tensors,
                                            const CAttentionWeight<T>* weights,
                                            const HelperIRPara<T>* helper_weights,
                                            cudaStream_t stream);
  
    /// fp16, fp16, fp16  chw optimize v2
    void __forward_half_linear_v2_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                         const std::vector<fastertransformer::Tensor>* input_tensors,
                                         const CAttentionWeight<T>* weights);
    void __forward_magic_half_linear_v2_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                               const std::vector<fastertransformer::Tensor>* input_tensors,
                                               const CAttentionWeight<T>* weights,
                                               const HelperIRPara<T>* helper_weights,
                                               cudaStream_t stream);

    /// fp16, fp16, fp16  AVOD chw optimize v2
    void __forward_half_linear_v2_AV_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                            const std::vector<fastertransformer::Tensor>* input_tensors,
                                            const CAttentionWeight<T>* weights);
    void __forward_magic_half_linear_v2_AV_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                                  const std::vector<fastertransformer::Tensor>* input_tensors,
                                                  const CAttentionWeight<T>* weights,
                                                  const HelperIRPara<T>* helper_weights,
                                                  cudaStream_t stream);

    /// int8, int8, int8 v2  AVOD
    void __forward_fake_int8_v2_AV_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                       const std::vector<fastertransformer::Tensor>* input_tensors,
                                       const CAttentionWeight<T>* weights);
    void __forward_magic_fake_int8_v2_AV_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                             const std::vector<fastertransformer::Tensor>* input_tensors,
                                             const CAttentionWeight<T>* weights,
                                             const HelperIRPara<T>* helper_weights,
                                             cudaStream_t stream);

protected:
    using BaseLayer::stream_;
    using BaseLayer::is_free_buffer_after_forward_;
    using BaseLayer::is_allocate_buffer_;
    using BaseLayer::cublas_wrapper_;
    using BaseLayer::allocator_;

    T* d_rp_norm_ = nullptr;  // [1, L, 3+1] expand one col
    int* d_img_shape_ = nullptr;
    float* d_pc_range_ = nullptr;   // attribute  [6]
    T* d_l2i_norm_ = nullptr;  // [NC,4,4]
    T* d_rp_matmuled_ = nullptr;    // [1, NC, 4, L]

    T* d_rpc_norm_ = nullptr;                  // [1, NC, L, 2]
    T* d_fs_output_ = nullptr;                 // [1, Ch, L, NC, 1, 4]
    T* d_attention_weights_output_ = nullptr;  // [L, 24]
    T* d_reduce_output_ = nullptr;             // [L, Ch]

    uint8_t* d_fs_mask_ = nullptr;

    /// pos encoder
    T* d_pencoder_bufs_0_ = nullptr;  //  [L, 3]

    T* d_pencoder_bufs_1_ = nullptr;  //  [L, embed_dims]
    T* d_pencoder_bufs_[2];

public:
    SVCrossAttentionLayer(size_t max_batch_size,
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
                          bool sparse = false,
                          int int8_mode = 0);

    SVCrossAttentionLayer(SVCrossAttentionLayer<T> const& layer);

    ~SVCrossAttentionLayer();

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const CAttentionWeight<T>* weights,
                 cudaStream_t stream);

    void forward_magic(std::vector<fastertransformer::Tensor>* output_tensors,
                       const std::vector<fastertransformer::Tensor>* input_tensors,
                       const CAttentionWeight<T>* weights,
                       const HelperIRPara<T>* hweights,
                       cudaStream_t stream);
};

}  // namespace fastertransformer
