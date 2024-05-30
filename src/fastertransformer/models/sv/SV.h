/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#pragma once

#include <vector>

#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/SVClsLayer.h"
#include "src/fastertransformer/layers/SVRegUpdateLayer.h"

#include "src/fastertransformer/layers/attention_layers/SVCrossAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers/SVUnfusedAttentionLayer.h"
#include "src/fastertransformer/models/sv/SVWeight.h"

namespace fastertransformer {

template<typename T>
class SVTransformer: public BaseLayer {
    size_t max_batch_size_;
    size_t max_seq_len_;

    size_t embed_dim_;
    size_t head_num_;    // mha head num
    size_t inter_size_;  // FFN internal size
    size_t num_layer_;

    size_t num_cam_;  // sv cross attention
    size_t l2i_h_;
    size_t l2i_w_;
    int img_shape_[2];
    float pc_range_[6];

    bool with_cls_token_;
    int sm_;
    float q_scaling_;
    const SVWeight<T>* weights_;

    AttentionType attention_type_;

    /// inner use
    size_t head_dim_;  // each head size,  embed_dim / head_num
    size_t request_seq_len_;
    bool is_allocate_buffer_;

    /// layer
    SVUnfusedAttentionLayer<T>* svmh_attention_layer_;
    FfnLayer<T>* ffn_layer_;
    SVCrossAttentionLayer<T>* svcross_attention_layer_;
    SVRegUpdateLayer<T>* svreg_update_layer_;
    SVClsLayer<T>* svcls_layer_;

    void allocateBuffer();
    void freeBuffer();
    bool setSeqLenVec(size_t batch_size);
    void setDefaultMask(size_t batch_size);
    void initialize();

protected:
    // T* query_buf_ = nullptr;
    T* query_buf_pair_ = nullptr;

    T* mha_out_buf_ = nullptr;
    T* mha_norm_out_buf_ = nullptr;
    T* mha_norm_out_with_pos_buf_ = nullptr;

    T* ca_out_buf_ = nullptr;
    T* ca_pos_feat_buf_ = nullptr;

    T* ca_norm_out_buf_ = nullptr;

    T* ffn_out_buf_ = nullptr;

    T* ffn_norm_out_buf_ = nullptr;
    T* reg_out_buf_ = nullptr;

    T* reg_out_norm_buf_ = nullptr;
    T* cls_out_norm_buf_ = nullptr;

    T* mask_buf_ = nullptr;

    int* seq_len_vec_ = nullptr;

public:
    SVTransformer(size_t max_batch_size,
                  size_t max_seq_len_,
                  size_t embed_dim,
                  size_t head_num,
                  size_t inter_size,
                  size_t num_layer,
                  size_t num_cam,
                  size_t l2i_h,
                  size_t l2i_w,
                  int* img_shape,
                  float* pc_range,
                  bool with_cls_token,
                  int sm,
                  float q_scaling,
                  const SVWeight<T>* weights,
                  cudaStream_t stream,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator* allocator,
                  bool is_free_buffer_after_forward = false,
                  AttentionType attention_type = AttentionType::UNFUSED_MHA);

    SVTransformer(SVTransformer<T> const& vit_layer);

    ~SVTransformer();

    void
    forward(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors, const SVWeight<T>* weights);

    void forward(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors, cudaStream_t stream);
};

}  // namespace fastertransformer
