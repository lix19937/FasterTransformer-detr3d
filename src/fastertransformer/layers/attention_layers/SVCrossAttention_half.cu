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
void SVCrossAttentionLayer<half>::__forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                            const std::vector<fastertransformer::Tensor>* input_tensors,
                                            const CAttentionWeight<half>* weights)
{
    ///  all int8, kc/32hw32,  for nc=6 OR 4
    if (input_tensors->at(2).type == DataType::TYPE_INT8 && input_tensors->at(3).type == DataType::TYPE_INT8
        && input_tensors->at(4).type == DataType::TYPE_INT8) {
        if (num_cam_ == 4) {
            __forward_fake_int8_v2_AV_branch(output_tensors, input_tensors, weights);
        }
        else {
            __forward_fake_int8_v2_branch(output_tensors, input_tensors, weights);
        }
        return;
    }

    /// all fp16, klinear
    if (input_tensors->at(2).type == DataType::TYPE_FP16 && input_tensors->at(3).type == DataType::TYPE_FP16
        && input_tensors->at(4).type == DataType::TYPE_FP16) {
        if (num_cam_ == 4) {
            __forward_half_linear_v2_AV_branch(output_tensors, input_tensors, weights);
        }
        else {
            __forward_half_linear_v2_branch(output_tensors, input_tensors, weights);
        }
    }
}

template<>
void SVCrossAttentionLayer<half>::forward_magic(std::vector<fastertransformer::Tensor>* output_tensors,
                                                const std::vector<fastertransformer::Tensor>* input_tensors,
                                                const CAttentionWeight<half>* weights,
                                                const HelperIRPara<half>* helper_weights,
                                                cudaStream_t stream)
{
    ///  all int8, kc/32hw32 
    if (input_tensors->at(0).type == DataType::TYPE_INT8 && input_tensors->at(1).type == DataType::TYPE_INT8
        && input_tensors->at(2).type == DataType::TYPE_INT8) {
        if (num_cam_ == 4) {
            __forward_magic_fake_int8_v2_AV_branch(output_tensors, input_tensors, weights, helper_weights, stream);
        }
        else {
            __forward_magic_fake_int8_v2_branch(output_tensors, input_tensors, weights, helper_weights, stream);
        }
        return;
    }

    /// all fp16, klinear
    if (input_tensors->at(0).type == DataType::TYPE_FP16 && input_tensors->at(1).type == DataType::TYPE_FP16
        && input_tensors->at(2).type == DataType::TYPE_FP16) {
        if (num_cam_ == 4) {
            __forward_magic_half_linear_v2_AV_branch(output_tensors, input_tensors, weights, helper_weights, stream);
        }
        else {
            __forward_magic_half_linear_v2_branch(output_tensors, input_tensors, weights, helper_weights, stream);
        }
    }
}

}  // namespace fastertransformer
