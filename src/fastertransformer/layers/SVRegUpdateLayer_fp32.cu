/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "SVRegUpdateLayer.h"

#include "src/fastertransformer/kernels/svadd_bias_relu_kernels.h"

namespace fastertransformer {

namespace fp32 {

// tmp = reg_branches[lid](output) // [L, 8]
// assert reference_points.shape[-1] == 3 // [L, 3]
// new_reference_points = torch.zeros_like(reference_points)
// new_reference_points[...,  :2] = tmp[...,  :2] + inverse_sigmoid(reference_points[...,  :2])
// new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
// new_reference_points = new_reference_points.sigmoid()
__global__ void IsigmoidAddBiasKernel(float* out,
                                      const float* __restrict__ reg, /* map to `tmp`, [L, 8] */
                                      const float* __restrict__ rp,  /* map to `reference_points` [L, 3] */
                                      const float* __restrict__ bias,
                                      const int n,
                                      const int k)
{
    int col_idx = threadIdx.x; /* 0,1,2 */
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int rp_idx = row_idx * n + col_idx;

    const int reg_offset_map[]{0, 1, 4};

    const float eps = 1.e-5;
    float x = __ldg(&rp[rp_idx]);
    x = min(max(x, 0.f), 1.f);
    auto x1 = max(x, eps);
    auto x2 = max(1.f - x, eps);
    float a = __ldg(&reg[row_idx * k + reg_offset_map[col_idx]]) + __ldg(&bias[reg_offset_map[col_idx]]);

    out[rp_idx] = 1.f / (1.f + expf(-(log(x1 / x2) + a)));
}

// tmp = reg_branches[lid](output) // [L, 8]
// assert reference_points.shape[-1] == 3 // [L, 3]
// new_reference_points = torch.zeros_like(reference_points)
// new_reference_points[...,  :2] = tmp[...,  :2] + inverse_sigmoid(reference_points[...,  :2])
// new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
// new_reference_points = new_reference_points.sigmoid()
//
// if lid == 3:
//     tmp[..., :2]  = new_reference_points[...,  :2]*102.4 + (-51.2)
//     tmp[..., 4:5] = new_reference_points[..., 2:3]*8.0   + (-5.0)
//     out_coord = tmp
// ==>
// if lid == 3:
//     tmp[..., :2]  = (tmp[...,  :2]  + inverse_sigmoid(reference_points[...,  :2])).sigmoid()*102.4 + (-51.2)
//     tmp[..., 4:5] = (tmp[...,  4:5] + inverse_sigmoid(reference_points[...,  2:3])).sigmoid()*8.0   + (-5.0)
//     out_coord = tmp
__global__ void IsigmoidAddBiasKernelFused(float* out,                     /* [L, 8] */
                                           const float* __restrict__ in,   /* map to tmp but no bias */
                                           const float* __restrict__ rp,   /* map to reference_points */
                                           const float* __restrict__ bias, /* [1, 8] */
                                           const float* __restrict__ pc_range,
                                           const int n,
                                           const int k)
{
    int col_idx = threadIdx.x;  /// 0, 1, 2,...7
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int reg_idx = row_idx * k + col_idx; /* reg_idx_of_row_begin */

    // 2, 3, 5,6,7
    if (col_idx > 4 || col_idx == 2 || col_idx == 3) {
        out[reg_idx] = __ldg(&in[reg_idx]) + __ldg(&bias[col_idx]);
        return;
    }

    // 0, 1,  4 --> 0 1 2
    auto get_idx = [col_idx]() {
        switch (col_idx) {
            case 0:
            case 1:
                return col_idx;
            case 4:
                return 2;
            default:
                break;
        }
        return -1;
    };
    float a = __ldg(&in[reg_idx]) + __ldg(&bias[col_idx]);

    const float eps = 1.e-5;
    float x = __ldg(&rp[row_idx * n + get_idx()]);  // 0,1,2
    x = min(max(x, 0.f), 1.f);
    float x1 = max(x, eps);
    float x2 = max(1.f - x, eps);
    //////////////// svod
    // out[reg_idx] = col_idx < 2 ? (1.f / (1.f + expf(-(log(x1 / x2) + a)))) * 102.4f + (-51.2f) :
    //                              (1.f / (1.f + expf(-(log(x1 / x2) + a)))) * 8.f + (-5.f);
    //////////////// avod
    // out[reg_idx] = col_idx < 2 ? (1.f / (1.f + expf(-(log(x1 / x2) + a)))) * 40.4f + (-20.2f) :
    //                              (1.f / (1.f + expf(-(log(x1 / x2) + a)))) * 8.f + (-5.f);
    switch (col_idx) {
        case 0:
            out[reg_idx] = (1.f / (1.f + expf(-(log(x1 / x2) + a)))) * __ldg(&pc_range[3]) + __ldg(&pc_range[0]);
            break;

        case 1:
            out[reg_idx] = (1.f / (1.f + expf(-(log(x1 / x2) + a)))) * __ldg(&pc_range[4]) + __ldg(&pc_range[1]);
            break;

        case 2:
        default:
            out[reg_idx] = (1.f / (1.f + expf(-(log(x1 / x2) + a)))) * __ldg(&pc_range[5]) + __ldg(&pc_range[2]);
            break;
    }
}
}  // namespace fp32

// for _ in range(2):
//     reg_branch.append(Linear(embed_dims,embed_dims))
//     reg_branch.append(nn.ReLU())
// reg_branch.append(Linear(embed_dims,8))
void RegBranch(float* buf[],            /* fc1_out,  fc2_out */
               const float* ffn_ln_out, /* block of ffn_ln`s output  [L, embed_dims] */
               const RegBranchWeight<float>* weights,
               int m,   /* L */
               int k,   /* embed_dims */
               int ich, /* embed_dims */
               int och, /* 8, hard code*/
               cublasMMWrapper* cublas_wrapper,
               cudaStream_t stream)
{
    // [L, E_d] * [E_d, E_d] --> [L, E_d]    + [1, E_d] bias
    int n = ich;
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights->fc1.kernel, n, ffn_ln_out, k, buf[0], n, 1.f, 0.f);

    sv::invokeAddBiasRelu(buf[0], weights->fc1.bias, m, n, stream);

    // [L, E_d] * [E_d, E_d] --> [L, E_d]    + [1, E_d] bias
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights->fc2.kernel, n, buf[0], k, buf[1], n, 1.f, 0.f);

    sv::invokeAddBiasRelu(buf[1], weights->fc2.bias, m, n, stream);

    // [L, E_d] * [E_d, 8] --> [L, 8]
    n = och;
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights->fc3.kernel, n, buf[1], k, buf[2], n, 1.f, 0.f);
}

void Update(float* out,           /* [L, 1, 3] */
            const float* reg,     /* from  RegBranch but no bias [L, 1, 8] */
            const float* rp,      /* reference_points [L, 3] */
            const float* fc_bias, /* [1, 8] */
            const int m,          /* L */
            const int n,          /* 3 */
            const int k,          /* 8 */
            cudaStream_t stream)
{
    dim3 grid(1, 2);
    dim3 block(n, 256);

    fp32::IsigmoidAddBiasKernel<<<grid, block, 0, stream>>>(out, reg, rp, fc_bias, n, k);
}

void UpdateFused(float* out,            /* [L, 1, 8] */
                 const float* in,       /* from RegBranch but no bias [L, 1, 8] */
                 const float* rp,       /* reference_points [L, 3] */
                 const float* fc_bias,  /* [1, 8] */
                 const float* pc_range, /* [1, 6] const */
                 const int m,           /* L */
                 const int n,           /* 3 */
                 const int k,           /* 8 */
                 cudaStream_t stream)
{
    dim3 grid(1, 4);
    dim3 block(k, 256 / 2);

    fp32::IsigmoidAddBiasKernelFused<<<grid, block, 0, stream>>>(out, in, rp, fc_bias, pc_range, n, k);
}

template<>
void SVRegUpdateLayer<float>::__forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                        const std::vector<fastertransformer::Tensor>* input_tensors,
                                        const RegBranchWeight<float>* weights)
{
    /// OUT
    auto d_rp_new = (float*)output_tensors->at(0).data; /* [L, 1, 3] */

    /// IN
    auto in_ffn_ln_output = (const float*)input_tensors->at(0).data;  // [L, embed_dims]  add with query_pos
    auto in_rp = (const float*)input_tensors->at(1).data;             // [1, L, 3] reference_points
    const int in_rp_w = *input_tensors->at(1).shape.rbegin();         /* 3, map to reference_points.shape[-1] */

    RegBranch(fc_buf_,          /* buffer and last out */
              in_ffn_ln_output, /* [const in], block of ffn_ln`s output  [L, embed_dims] */
              weights,
              num_query_,    /* L */
              hidden_units_, /* embed_dims */
              hidden_units_, /* embed_dims */
              8,             /* 8 */
              cublas_wrapper_,
              stream_);

    Update(d_rp_new,          /* new rp  [L, 3] */
           fc_buf_[2],        /* from  RegBranch but no bias [L, 8] */
           in_rp,             /* reference_points [L, 3] */
           weights->fc3.bias, /* bias [1, 8] */
           num_query_,        /* L */
           in_rp_w,           /* 3 */
           8,
           stream_);
}

template<>
void SVRegUpdateLayer<float>::forward_fused(std::vector<fastertransformer::Tensor>* output_tensors,
                                            const std::vector<fastertransformer::Tensor>* input_tensors,
                                            const RegBranchWeight<float>* weights,
                                            cudaStream_t stream)
{
    stream_ = stream;

    /// OUT
    auto d_rp_new = (float*)output_tensors->at(0).data; /* [L, 1, 8] */
    int rp_new_w = output_tensors->at(0).shape.at(2);

    /// IN
    auto in_ffn_ln_output = (const float*)input_tensors->at(0).data;  // [L, embed_dims]  add with query_pos
    auto in_rp = (const float*)input_tensors->at(1).data;             // [1, L, 3] reference_points
    int rp_w = input_tensors->at(1).shape.at(2);

    float* inner_fc_buf[]{fc_buf_[0], fc_buf_[1], fc_buf_[0]};
    RegBranch(inner_fc_buf,
              in_ffn_ln_output, /* block of ffn_ln`s output  [L, embed_dims] */
              weights,
              num_query_,    /* L */
              hidden_units_, /* embed_dims */
              hidden_units_, /* embed_dims */
              rp_new_w,      /* 8 */
              cublas_wrapper_,
              stream_);

    UpdateFused(d_rp_new,          /* [L, 8] */
                fc_buf_[0],        /* from RegBranch but no bias [L, 8] */
                in_rp,             /* reference_points [L, 3] */
                weights->fc3.bias, /* bias [1, 8] */
                d_pc_range_,
                num_query_, /* L */
                rp_w,       /* 3 */
                rp_new_w,   /* 8 */
                stream_);

    sync_check_cuda_error();
}

}  // namespace fastertransformer
