/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "SVClsLayer.h"

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/svadd_bias_relu_kernels.h"

namespace fastertransformer {

/// __half22float2
// (0): Linear(in_features=256, out_features=256, bias=True)
// (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
// (2): ReLU(inplace=True)
// (3): Linear(in_features=256, out_features=256, bias=True)
// (4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
// (5): ReLU(inplace=True)
// (6): Linear(in_features=256, out_features=8, bias=True)

void ClsBranch(half* buf[],     // fc1_out,  fc2_out
               float* cls_out,  // block of ffn_ln`s output  [L, embed_dims]
               const half* in,
               const ClsBranchWeight<half>* weights,
               int m,   /* L */
               int k,   /* embed_dims */
               int ich, /* embed_dims */
               int och, /* class_num 8 */
               cublasMMWrapper* cublas_wrapper,
               cudaStream_t stream)
{
    // [L, E_d] * [E_d, E_d] --> [L, E_d]    + [1, E_d] bias
    int n = ich;
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights->fc1.kernel, n, in, k, buf[0], n, 1.f, 0.f);

    svpost_relu::invokeAddBiasLayernorm(
        buf[0], weights->fc1.bias, weights->ln1.kernel, weights->ln1.bias, m, n, stream);

    // [L, E_d] * [E_d, E_d] --> [L, E_d]    + [1, E_d] bias
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights->fc2.kernel, n, buf[0], k, buf[1], n, 1.f, 0.f);

    svpost_relu::invokeAddBiasLayernorm(
        buf[1], weights->fc2.bias, weights->ln2.kernel, weights->ln2.bias, m, n, stream);

    // [L, E_d] * [E_d, class_num] --> [1, L, class_num]
    n = och;
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights->fc3.kernel, n, buf[1], k, buf[0], n, 1.f, 0.f);

    sv::invokeAddBias(cls_out, buf[0], weights->fc3.bias, m, n, stream);
}

template<>
void SVClsLayer<half>::__forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                 const std::vector<fastertransformer::Tensor>* input_tensors,
                                 const ClsBranchWeight<half>* weights)
{
    /// OUT
    auto d_cls_out = (float*)output_tensors->at(0).data; /* [1, L, num_classes] */
    const int num_classes = *output_tensors->at(0).shape.rbegin();

    /// IN
    auto d_in_ffn_ln_output = (const half*)input_tensors->at(0).data; /* [1, L, embed_dims]  add with query_pos */

    ClsBranch(fc_buf_,
              d_cls_out,
              d_in_ffn_ln_output, /* block of ffn_ln`s output  [L, embed_dims] */
              weights,
              num_query_,    /* L */
              hidden_units_, /* embed_dims */
              hidden_units_, /* embed_dims */
              num_classes,   /* 5 or 8 */
              cublas_wrapper_,
              stream_);
}

}  // namespace fastertransformer
