/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "SVCrossAttentionLayer.h"

namespace fastertransformer {

template<>
void SVCrossAttentionLayer<float>::__forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                             const std::vector<fastertransformer::Tensor>* input_tensors,
                                             const CAttentionWeight<float>* weights)
{
    if (num_cam_ == 4) {
        __forward_fp32_AV_branch(output_tensors, input_tensors, weights);
    }
    else if (num_cam_ == 6 || num_cam_ == 12) {
        __forward_fp32_SV_branch(output_tensors, input_tensors, weights);
    }
}

template<>
void SVCrossAttentionLayer<float>::forward_magic(std::vector<fastertransformer::Tensor>* output_tensors,
                                                 const std::vector<fastertransformer::Tensor>* input_tensors,
                                                 const CAttentionWeight<float>* weights,
                                                 const HelperIRPara<float>* helper_weights,
                                                 cudaStream_t stream)
{
    if (num_cam_ == 4) {
        __forward_magic_fp32_AV_branch(output_tensors, input_tensors, weights, helper_weights, stream);
    }
    else if (num_cam_ == 6 || num_cam_ == 12) {
        __forward_magic_fp32_SV_branch(output_tensors, input_tensors, weights, helper_weights, stream);
    }
}

}  // namespace fastertransformer
