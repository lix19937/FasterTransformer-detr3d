/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "SVCrossAttentionLayer.h"

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/models/sv/helper_file.h"


namespace fastertransformer {

namespace fp32 {

__forceinline__ __device__ bool within_bounds_2d(const int h, const int w, const int H, const int W)
{
    return h >= 0 && w >= 0 && h < H && w < W;
}

__global__ void TransposeAndNormKernel(float* __restrict__ out,
                                       const float* __restrict__ in,
                                       const float* __restrict__ range,
                                       const int height,
                                       const int width)
{
    const int BLOCK_DIM = 32;
    __shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (xIndex < width && yIndex < height) {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = __ldg(&in[index_in]);
    }

    __syncthreads();

    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if (xIndex < height && yIndex < width) {
        unsigned int index_out = yIndex * height + xIndex;
        out[index_out] = block[threadIdx.x][threadIdx.y] * __ldg(&range[yIndex + 3]) + __ldg(&range[yIndex]);
    }
    else if (xIndex < height && yIndex == width) {
        unsigned int index_out = yIndex * height + xIndex;
        out[index_out] = 1;
    }
}

__global__ void DivKernel(float* __restrict__ out,
                          const float* __restrict__ in /* net IN  lidar2img matr*/,
                          const int* __restrict__ dv /* net IN shape*/,
                          const int w)
{
    auto yIndex = threadIdx.y;
    auto idx = yIndex * w + threadIdx.x;
    auto res = yIndex % w;
    if (res == 0)
        out[idx] = __ldg(&in[idx]) / __ldg(&dv[1]);
    else if (res == 1)
        out[idx] = __ldg(&in[idx]) / __ldg(&dv[0]);
    else
        out[idx] = __ldg(&in[idx]);
}

__global__ void DivKernel_w4(float* __restrict__ out,
                             const float* __restrict__ in /* net IN  lidar2img matr*/,
                             const int* __restrict__ dv /* net IN shape*/)
{
    auto yIndex = threadIdx.y;
    auto idx = (yIndex << 2) + threadIdx.x;  // / 4
    auto res = yIndex & 3;                   // % 4
    if (res == 0)
        out[idx] = __ldg(&in[idx]) / __ldg(&dv[1]);
    else if (res == 1)
        out[idx] = __ldg(&in[idx]) / __ldg(&dv[0]);
    else
        out[idx] = __ldg(&in[idx]);
}

__global__ void ReferencePointsCamAndMaskPermuteKernel(uint8_t* __restrict__ mask,
                                                       float* __restrict__ rpc,
                                                       const float* __restrict__ rpc_matmuled,
                                                       const unsigned int NC,
                                                       const unsigned int w,
                                                       const unsigned int L)
{
    int l_idx = threadIdx.x;  // map to L, col direct
    int nc_idx = blockIdx.x;  // map to NC, row direct

    /// 4 is fixed, because width of reference_points is 3, and exp 1 dim
    /// for [1, NC, 4, L] of rpc_matmuled   [:,:,2:3,:]
    unsigned int dim_acc[]{0, 4 * L, L, 1};
    auto idx_23 = nc_idx * dim_acc[1] + 2 * dim_acc[2] + l_idx;
    auto idx_01 = nc_idx * dim_acc[1] + 0 * dim_acc[2] + l_idx;
    auto idx_12 = nc_idx * dim_acc[1] + 1 * dim_acc[2] + l_idx;

    auto reference_points_cam_23 = __ldg(&rpc_matmuled[idx_23]);
    auto reference_points_cam_01 = __ldg(&rpc_matmuled[idx_01]);
    auto reference_points_cam_12 = __ldg(&rpc_matmuled[idx_12]);

    /// from  [1, NC, 2, L] permute to [1, NC, L, 2], rpc restore layout
    unsigned int dim_acc_d[]{0, L * 2, 2, 1};
    auto idx_01_d = nc_idx * dim_acc_d[1] + l_idx * dim_acc_d[2];
    auto idx_12_d = nc_idx * dim_acc_d[1] + l_idx * dim_acc_d[2] + 1;

    /// from [1, NC, L, 1] permute to [1, L, NC, 1], mask restore layout
    unsigned int dim_acc_e[]{0, NC, 1, 1};
    auto idx_01_e = l_idx * dim_acc_e[1] + nc_idx * dim_acc_e[2];

    if (reference_points_cam_23 > 1.e-2) {
        auto m = reference_points_cam_01 / reference_points_cam_23;
        auto n = reference_points_cam_12 / reference_points_cam_23;

        if (m > 2.f)
            m = 2.f;
        if (n > 2.f)
            n = 2.f;
        if (m < -1.f)
            m = -1.f;
        if (n < -1.f)
            n = -1.f;

        m = m + m - 1.f;  //(m - 0.5) * 2;
        n = n + n - 1.f;  //(n - 0.5) * 2;
        rpc[idx_01_d] = m;
        rpc[idx_12_d] = n;

        if (m < 1. && m > -1. && n < 1. && n > -1.) {
            mask[idx_01_e] = 1;  // true
        }
        else {
            mask[idx_01_e] = 0;  // false
        }
    }
    else {
        rpc[idx_01_d] = -3.f;  //(-1 - 0.5) * 2;
        rpc[idx_12_d] = -3.f;  //(-1 - 0.5) * 2;
        mask[idx_01_e] = 0;
    }
}

__forceinline__ __device__ float grid_sampler_compute_source_index(const float coord, const int size)
{
    return ((coord + 1.f) * size - 1) * 0.5f;
}

__global__ void
grid_sampler_2d_forward_kernel(const int nthreads,
                               const float* __restrict__ input,
                               const int inp_N,
                               const int inp_C,
                               const int inp_H,
                               const int inp_W,
                               const float* __restrict__ grid,
                               const int grid_H,
                               const int grid_W,
                               float* __restrict__ output /*N = inp_N, C = inp_C, H = grid_H, W = grid_W*/,
                               const int stack_sz,
                               const int stack_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nthreads)
        return;

    int C = inp_C;
    int out_H = grid_H;
    int out_W = grid_W;

    int inp_sW = 1;
    int inp_sH = inp_sW * inp_W;
    int inp_sC = inp_sH * inp_H;
    int inp_sN = inp_sC * inp_C;

    int grid_sCoor = 1;
    int grid_sW = grid_sCoor * 2;
    int grid_sH = grid_sW * grid_W;
    int grid_sN = grid_sH * grid_H;  /// grid size : N C H 2

    int out_sW = 1;
    int out_sH = grid_W;  ///  eq: out_sW * grid_W
    int out_sC = out_sH * grid_H;
    int out_sN = out_sC * inp_C;

    int CH = C * out_H;

    //#pragma unroll
    // for (; i < nthreads; i += blockDim.x * gridDim.x)
    {
        const int w = i % out_W;
        const int h = (i / out_W) % out_H;
        const int n = i / (out_H * out_W);
        const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

        // get the corresponding input x, y co-ordinates from grid
        float ix = __ldg(&grid[grid_offset]);
        float iy = __ldg(&grid[grid_offset + grid_sCoor]);
        ix = grid_sampler_compute_source_index(ix, inp_W);
        iy = grid_sampler_compute_source_index(iy, inp_H);

        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw = static_cast<int>(::floor(ix));
        int iy_nw = static_cast<int>(::floor(iy));

        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);

        /// calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = input + n * inp_sN;
        /// auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
        auto in_offset = n * out_sN + h * out_sH + w * out_sW;

        float sum;

        // int offset1 = iy_nw * inp_sH + ix_nw * inp_sW;
        // int offset2 = iy_ne * inp_sH + ix_ne * inp_sW;
        // int offset3 = iy_sw * inp_sH + ix_sw * inp_sW;
        // int offset4 = iy_se * inp_sH + ix_se * inp_sW;

        int mul1 = iy_nw * inp_sH;  // iy_nw == iy_ne
        int mul2 = ix_nw * inp_sW;  // ix_nw == ix_sw
        int mul3 = iy_sw * inp_sH;  // iy_sw == iy_se
        int mul4 = ix_ne * inp_sW;  // ix_ne == ix_se

        int offset1 = mul1 + mul2;
        int offset2 = mul1 + mul4;
        int offset3 = mul3 + mul2;
        int offset4 = mul3 + mul4;

#pragma unroll
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, in_offset += out_sC) {
            sum = 0.f;
            if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                sum += inp_ptr_NC[offset1] * nw;
            }
            if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                sum += inp_ptr_NC[offset2] * ne;
            }
            if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                sum += inp_ptr_NC[offset3] * sw;
            }
            if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                sum += inp_ptr_NC[offset4] * se;
            }

            ///             inp_N, inp_C, grid_H, grid_W  [6, 256, L, 1],
            /// so simplify to [nc, 256, L] --> [nc, 256x L]
            {
                int c_idx = in_offset / CH;
                int hw_idx = in_offset % CH;
                int out_offset = (hw_idx * inp_N + c_idx) * stack_sz + stack_idx;
                output[out_offset] = sum;
            }
        }
    }
}

__forceinline__ __device__ void convertchw_maxpool2(
    const size_t idx, const size_t area, const size_t w, const float* __restrict__ input, float* __restrict__ value)
{
    auto AREA = area << 2;  /// 4 * area, W = 2 * w, before pool
    auto raw_idx = (idx / area * AREA) + ((idx % area / w) << 1) * (w << 1) + (idx % area % w << 1);

    *value = __ldg(&input[raw_idx]);
}

__global__ void grid_sampler_2d_forward_kernel_n6c256gh512gw1_maxpool(
    const int nthreads,
    const float* __restrict__ input,
    const int inp_N,
    const int inp_C,  // 256
    const int inp_H,
    const int inp_W,
    const float* __restrict__ grid /* __restrict__ reference_points_cam after norm */,
    const int grid_H,  // 512
    const int grid_W,  // 1
    float* __restrict__ output /* N = inp_N, C = inp_C, H = grid_H, W = grid_W*/,
    const int stack_sz,
    const int stack_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nthreads)
        return;

    int C = inp_C;
    int out_H = grid_H;
    int out_W = grid_W;

    int inp_sW = 1;
    int inp_sH = inp_sW * inp_W;
    int inp_sC = inp_sH * inp_H;
    int inp_sN = inp_sC * inp_C;

    int grid_sCoor = 1;
    int grid_sW = grid_sCoor * 2;
    int grid_sH = grid_sW * grid_W;
    int grid_sN = grid_sH * grid_H; /* grid size : N C H 2 */

    int out_sW = 1;
    int out_sH = grid_W;  ///  eq: out_sW * grid_W
    int out_sC = out_sH * grid_H;
    int out_sN = out_sC * inp_C;

    int CH = C * out_H;

    //#pragma unroll
    // for (; i < nthreads; i += blockDim.x * gridDim.x)
    {
        const int w = i % out_W;
        const int h = (i / out_W) % out_H;
        const int n = i / (out_H * out_W);
        const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

        // get the corresponding input x, y co-ordinates from grid
        float ix = __ldg(&grid[grid_offset]);
        float iy = __ldg(&grid[grid_offset + grid_sCoor]);
        ix = grid_sampler_compute_source_index(ix, inp_W);
        iy = grid_sampler_compute_source_index(iy, inp_H);

        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw = static_cast<int>(::floor(ix));
        int iy_nw = static_cast<int>(::floor(iy));

        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);

        /// calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = n * inp_sN;
        /// auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
        auto in_offset = n * out_sN + h * out_sH + w * out_sW;

        // int offset1 = iy_nw * inp_sH + ix_nw * inp_sW;
        // int offset2 = iy_ne * inp_sH + ix_ne * inp_sW;
        // int offset3 = iy_sw * inp_sH + ix_sw * inp_sW;
        // int offset4 = iy_se * inp_sH + ix_se * inp_sW;

        int mul1 = iy_nw * inp_sH;  // iy_nw == iy_ne
        int mul2 = ix_nw * inp_sW;  // ix_nw == ix_sw
        int mul3 = iy_sw * inp_sH;  // iy_sw == iy_se
        int mul4 = ix_ne * inp_sW;  // ix_ne == ix_se

        int offset1 = mul1 + mul2;
        int offset2 = mul1 + mul4;
        int offset3 = mul3 + mul2;
        int offset4 = mul3 + mul4;
        const size_t area = inp_H * inp_W;
        float sum, value;

#pragma unroll
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, in_offset += out_sC) {
            sum = 0.f;
            if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                convertchw_maxpool2(inp_ptr_NC + offset1, area, inp_W, input, &value);
                sum += value * nw;
            }
            if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                convertchw_maxpool2(inp_ptr_NC + offset2, area, inp_W, input, &value);
                sum += value * ne;
            }
            if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                convertchw_maxpool2(inp_ptr_NC + offset3, area, inp_W, input, &value);
                sum += value * sw;
            }
            if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                convertchw_maxpool2(inp_ptr_NC + offset4, area, inp_W, input, &value);
                sum += value * se;
            }

            ///             inp_N, inp_C, grid_H, grid_W  [6, 256, L, 1],
            /// so simplify to [nc, 256, L] --> [nc, 256x L]
            {
                int c_idx = in_offset / CH;
                int hw_idx = in_offset % CH;
                int out_offset = (hw_idx * inp_N + c_idx) * stack_sz + stack_idx;
                output[out_offset] = sum;
            }
        }
    }
}

__global__ void AddBiasAttentionWeightsSigmoidMaskKernel(float* __restrict__ out,
                                                         const float* __restrict__ attention_weights,
                                                         const float* __restrict__ fc_attention_bias,
                                                         const uint8_t* __restrict__ fs_mask,
                                                         const unsigned int bias_len)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto fs_mask_idx = idx >> 2;     //  / 4
    auto bias_idx = idx % bias_len;  // [nc,1,4]

    if (__ldg(&fs_mask[fs_mask_idx]) == 0) {
        out[idx] = 0.f;
    }
    else {
        float x = __ldg(&attention_weights[idx]) + __ldg(&fc_attention_bias[bias_idx]);
        out[idx] = 1.f / (1.f + exp(-x));
    }
}

__global__ void MulAndReducesumKernel(float* __restrict__ reduce_output,
                                      const float* __restrict__ fs_output,
                                      const float* __restrict__ mul_out,
                                      const int mul_out_dims_length,
                                      const int scale,
                                      const int reduce_num)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto m_idx = idx % mul_out_dims_length;
    auto tid = threadIdx.x;
    extern __shared__ float shm[];

    shm[tid] = __ldg(&fs_output[idx]) * __ldg(&mul_out[m_idx]);
    __syncthreads();

    // const int reduce_num = 24; //  nc * 1 * 4
#pragma unroll
    for (unsigned int s = reduce_num >> 1; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int offset = 0; offset < blockDim.x; offset += reduce_num) {
                shm[tid + offset] += shm[tid + s + offset];
            }
        }
        __syncthreads();
    }

    auto bid = blockIdx.x * scale;
#pragma unroll
    for (unsigned int i = 0; i < blockDim.x; i += reduce_num) {
        if (tid == i) {
            // transpose, [ch,  L] --> [L, ch]
            unsigned int out_idx = bid + i / reduce_num;
            unsigned int r = out_idx >> 9;
            unsigned int c = out_idx & 511;
            reduce_output[(c << 8) + r] = shm[tid] + shm[tid + 2];
        }
    }
}

__global__ void IsigmoidAddBiasKernel(float* out, const float* __restrict__ rp /* reference_points */)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    const float eps = 1.e-5;

    float x = __ldg(&rp[idx]);
    x = min(max(x, 0.f), 1.f);
    auto x1 = max(x, eps);
    auto x2 = max(1.f - x, eps);
    out[idx] = log(x1 / x2);
}

void ReferencePointsNorm(float* rp_norm,        /* reference_points after norm [L, 3](memory is L*4)  -->  [4, L] */
                         const float* rp,       /* reference_points`s shape B=1, from IN */
                         const float* pc_range, /* from Attri */
                         const int in_h,        /* L */
                         const int in_w,        /* 3 */
                         cudaStream_t stream)
{
    const int block_w = 32, block_h = 32;
    dim3 grid((in_w + block_w - 1) / block_w, (in_h + block_h - 1) / block_h);
    dim3 block(block_w, block_h);
    TransposeAndNormKernel<<<grid, block, 0, stream>>>(rp_norm, rp, pc_range, in_h, in_w);
}

void L2IDivImgShpe(float* l2i_norm,  /* lidar2img/img_shapes; [ NC, 4, 4] */
                   const float* l2i, /* lidar2img  ; [NC*4, 4], from IN */
                   const int* shape, /* img_shapes ; [1, 4] { B[1], B[0], 1, 1 }, use 2, from IN */
                   const int nc,     /* nc */
                   const int h,      /* 4 */
                   const int w,      /* 4 */
                   cudaStream_t stream)
{
    dim3 block(w, nc * h);
    if (w == 4) {
        DivKernel_w4<<<1, block, 0, stream>>>(l2i_norm, l2i, shape);
    }
    else {
        DivKernel<<<1, block, 0, stream>>>(l2i_norm, l2i, shape, w);
    }
}

// [NC*4, 4] * [4, L] -->  [NC*4, L] or [1, NC, 4, L]  rpc_matmuled = l2i_norm * rp_norm
void L2IxReferencePoints(float* rpc_matmuled,   /* reference_points_cam after rp matmul; [1, NC, 4, L] */
                         const float* l2i_norm, /* lidar2img norm, which has div img_shapes; [NC*4, 4] */
                         const float* rp_norm,  /* rp has norm; [4, L] */
                         int m,                 /* nc * 4 */
                         int k,                 /* 4 */
                         int n,                 /* l */
                         cublasMMWrapper* cublas_wrapper,
                         cudaStream_t stream)
{
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, rp_norm, n, l2i_norm, k, rpc_matmuled, n, 1.f, 0.f);
}

void ReferencePointsCamAndMask(uint8_t* mask,             /* mask, last out; [1, L, NC, 1] */
                               float* rpc_norm,           /* reference_points_cam after norm; [1, NC, L, 2] */
                               const float* rpc_matmuled, /* reference_points_cam after torch.matmul; [1, NC, 4, L]*/
                               const unsigned int nc,
                               const unsigned int w, /* 4 */
                               const unsigned int l,
                               cudaStream_t stream)
{
    /// here we make sure L < 1024, and w == 4, Here will be improve
    dim3 grid(nc, 1);
    dim3 block(l, 1);
    ReferencePointsCamAndMaskPermuteKernel<<<grid, block, 0, stream>>>(mask, rpc_norm, rpc_matmuled, nc, w, l);
}

void BatchedBilinearGridSample(float* sampled_feats,  /* sampled_feats, last out; [CH, L, NC, 1, 4] */
                               const float* rpc_norm, /* reference_points_cam after norm; [1, NC, L, 2] */
                               const float* mlvl_feats[],
                               const std::vector<std::vector<size_t>>& mlvl_feats_dims, /* [4]; [ NC, Ch, _, _] */
                               const int seq_len,                                       /* num_cam, num_query */
                               cudaStream_t stream)
{
    const int inp_N = mlvl_feats_dims[0][0];
    const int inp_C = mlvl_feats_dims[0][1];

    const int grid_H = seq_len;
    const int grid_W = 1;              /// here we make sure grid_W == 1 !!!
    const int count = inp_N * grid_H;  /// inp_N * grid_H * grid_W
    const int stack_sz = mlvl_feats_dims.size();

    dim3 block(512 / 2);
    dim3 grid((count + block.x - 1) / block.x);

    for (int i = 0; i < stack_sz; ++i) {
        const int inp_H = mlvl_feats_dims[i][2];
        const int inp_W = mlvl_feats_dims[i][3];

        grid_sampler_2d_forward_kernel<<<grid, block, 0, stream>>>(count,
                                                                   mlvl_feats[i],
                                                                   inp_N,
                                                                   inp_C,
                                                                   inp_H,
                                                                   inp_W,
                                                                   rpc_norm,
                                                                   grid_H,
                                                                   grid_W,
                                                                   sampled_feats,
                                                                   4,  // stack_sz,
                                                                   i);
        /// last loop, then add maxpool
        if (i + 1 == stack_sz) {
            grid_sampler_2d_forward_kernel_n6c256gh512gw1_maxpool<<<grid, block, 0, stream>>>(count,
                                                                                              mlvl_feats[i],
                                                                                              inp_N,
                                                                                              inp_C,
                                                                                              inp_H / 2,
                                                                                              inp_W / 2,
                                                                                              rpc_norm,
                                                                                              grid_H,
                                                                                              grid_W,
                                                                                              sampled_feats,
                                                                                              4,  // stack_sz,
                                                                                              i + 1);
        }
    }
}

// [seq_len, embed_dims] * [embed_dims, 24] --> [seq_len, 24] -> [seq_len, nc, 1, 4]
void AttentionWeightsFc(float* attention_weights,
                        const float* query_embbed, /* attention_weights = self.attention_weights(query) */
                        const float* fc_attention_weights,
                        const int m,
                        const int k,
                        const int n,
                        cublasMMWrapper* cublas_wrapper,
                        cudaStream_t stream)
{
    cublas_wrapper->Gemm(
        CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, fc_attention_weights, n, query_embbed, k, attention_weights, n, 1.f, 0.f);
}

void TwoMulReduceSum(float* reduce_output, /* last output */
                     float* buf,
                     const float* attention_weights /* from AttentionWeightsFc output */,
                     const float* fc_attention_bias, /* AttentionWeightsFc which bias here  */
                     const float* fs_output,
                     const uint8_t* fs_mask,
                     const int ch,
                     const int l,
                     const int nc,
                     const int m,
                     const int k,
                     cudaStream_t stream)
{
    dim3 block(l);
    dim3 grid(nc * m * k);

    AddBiasAttentionWeightsSigmoidMaskKernel<<<grid, block, 0, stream>>>(
        buf, attention_weights, fc_attention_bias, fs_mask, grid.x);

    {
        const int scale = 4;
        auto block_w = nc * m * k;
        dim3 block(block_w * scale);
        dim3 grid(ch * l / scale);
        MulAndReducesumKernel<<<grid, block, block.x * sizeof(float), stream>>>(
            reduce_output, fs_output, buf, l * block_w, scale, block_w);
    }
}

void OutputProjFc(float* output,
                  const float* reduce_output,
                  const float* output_proj_weight,
                  const int m,
                  const int k,
                  const int n,
                  cublasMMWrapper* cublas_wrapper,
                  cudaStream_t stream)
{
    cublas_wrapper->Gemm(
        CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, output_proj_weight, n, reduce_output, k, output, n, 1.f, 0.f);
}

void PositionEncoder(float* pos_feat, /* [seq_len, embed_dims] */
                     float* buf[],
                     const float* rp, /* [seq_len, 3] */
                     const float* weights[],
                     int m, /* seq_len */
                     int k, /* 3 */
                     int n, /* embed_dims */
                     cublasMMWrapper* cublas_wrapper,
                     cudaStream_t stream)
{
    IsigmoidAddBiasKernel<<<k, m, 0, stream>>>(buf[0], rp);

    // [L, 3] * [3, E_d] --> [L, E_d]   + [1, E_d] bias
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights[0], n, buf[0], k, buf[1], n, 1.f, 0.f);

    // T* out, const T* bias, const T* gamma, const T* beta, int m, int n, cudaStream_t stream
    svpost_relu::invokeAddBiasLayernorm(buf[1], weights[1], weights[2], weights[3], m, n, stream);

    // [L, E_d] * [E_d, E_d] --> [L, E_d]  + [1, E_d] bias
    k = n;
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights[4], n, buf[1], k, pos_feat, n, 1.f, 0.f);

    svpost_relu::invokeAddBiasLayernorm(pos_feat, weights[5], weights[6], weights[7], m, n, stream);
}
}  // namespace fp32

template<>
void SVCrossAttentionLayer<float>::__forward_fp32_SV_branch(std::vector<fastertransformer::Tensor>* output_tensors,
                                                            const std::vector<fastertransformer::Tensor>* input_tensors,
                                                            const CAttentionWeight<float>* weights)
{
    const int nc = num_cam_, seq_len = num_query_, embed_dims = hidden_units_, ch = input_tensors->at(2).shape[1];

    /// OUT
    auto d_output_prj = (float*)output_tensors->at(0).data; /* [L, embed_dims] */
    auto d_pos_feat = (float*)output_tensors->at(1).data;   /* [L, embed_dims] */

    /// IN
    auto in_query_embbed = (const float*)input_tensors->at(0).data; /* [L, embed_dims]  has added with query_pos */
    auto in_rp = (const float*)input_tensors->at(1).data;           /* [1, L, 3] reference_points */
    const float* in_x[]{(const float*)input_tensors->at(2).data,
                        (const float*)input_tensors->at(3).data,
                        (const float*)input_tensors->at(4).data};
    const std::vector<std::vector<size_t>> in_x_shapes{
        input_tensors->at(2).shape, input_tensors->at(3).shape, input_tensors->at(4).shape}; /* [NC, Ch, _, _] */

    auto in_l2i = (const float*)input_tensors->at(5).data; /* [1, NC, 4, 4] */

    /// wieghts
    const auto w_fc_attention_weight = weights->attention_weights.kernel;
    const auto w_fc_attention_bias = weights->attention_weights.bias;
    const auto w_output_proj_weight = weights->output_proj.kernel;
    const float* w_pencoder_weights[]{weights->position_encoder_fc1.kernel,
                                      weights->position_encoder_fc1.bias,
                                      weights->position_encoder_ln1.kernel,
                                      weights->position_encoder_ln1.bias,
                                      weights->position_encoder_fc2.kernel,
                                      weights->position_encoder_fc2.bias,
                                      weights->position_encoder_ln2.kernel,
                                      weights->position_encoder_ln2.bias};

    fp32::ReferencePointsNorm(d_rp_norm_,                /* rp after norm; [1, 4, num_query] */
                              in_rp,                     /* rp from node in const; [1, num_query, 3] */
                              (const float*)d_pc_range_, /* from node attribute */
                              seq_len,
                              3, /* ReferencePoints width 3  */
                              stream_);

    fp32::L2IDivImgShpe(d_l2i_norm_,  /* [nc, 4, 4] */
                        in_l2i,       /* from node in; [nc, 4, 4] */
                        d_img_shape_, /* from node in; [1, 2] */
                        nc,
                        4, /* matrix h  */
                        4, /* matrix w */
                        stream_);

    fp32::L2IxReferencePoints((float*)d_rp_matmuled_,    /* rp after matmul; [1, NC, 4, L] */
                              (const float*)d_l2i_norm_, /* lidar2img, has div img_shapes; [1, 6, 4, 4] */
                              (const float*)d_rp_norm_,  /* rp after norm; [4, L] */
                              nc * 4,
                              4,
                              seq_len,
                              cublas_wrapper_,
                              stream_);

    fp32::ReferencePointsCamAndMask(d_fs_mask_,  /* last out; [1, L, NC, 1] */
                                    d_rpc_norm_, /* rpc after norm which as grid sample in;  [1, NC, L, 2] */
                                    (const float*)d_rp_matmuled_, /* [1, NC, 4, L] */
                                    nc,
                                    4u,
                                    seq_len,
                                    stream_);

    fp32::BatchedBilinearGridSample(d_fs_output_, /* last out; [1, Ch, L, NC, 1, 4] */
                                    (const float*)d_rpc_norm_,
                                    in_x,
                                    in_x_shapes,
                                    seq_len,
                                    stream_);

    fp32::AttentionWeightsFc(d_attention_weights_output_, /* output; [L, 24] , no bias*/
                             in_query_embbed,             /* [L, embed_dims] */
                             w_fc_attention_weight,       /* [embed_dims, 24] */
                             seq_len,
                             embed_dims,
                             nc * 4,
                             cublas_wrapper_,
                             stream_);

    fp32::TwoMulReduceSum(d_reduce_output_,            /* output; [L, ch] */
                          d_pencoder_bufs_1_,          /* buffer space [L, nc*1*4] */
                          d_attention_weights_output_, /* from AttentionWeightsFc output */
                          w_fc_attention_bias,         /* AttentionWeightsFc which bias here [1, nc*1*4] */
                          (const float*)d_fs_output_,
                          (const uint8_t*)d_fs_mask_,
                          ch,
                          seq_len,
                          nc,
                          1,
                          4,
                          stream_);

    fp32::OutputProjFc(d_output_prj, /* last output;  [L, embed_dims] */
                       (const float*)d_reduce_output_,
                       w_output_proj_weight, /* [embed_dims, embed_dims]  */
                       seq_len,
                       embed_dims,
                       embed_dims,
                       cublas_wrapper_,
                       stream_);

    fp32::PositionEncoder(d_pos_feat,       /* last output;  [L, embed_dims] */
                          d_pencoder_bufs_, /* tmp space */
                          in_rp,            /* const input  */
                          w_pencoder_weights,
                          seq_len,
                          3,
                          embed_dims,
                          cublas_wrapper_,
                          stream_);
}

template<>
void SVCrossAttentionLayer<float>::__forward_magic_fp32_SV_branch(
    std::vector<fastertransformer::Tensor>* output_tensors,
    const std::vector<fastertransformer::Tensor>* input_tensors,
    const CAttentionWeight<float>* weights,
    const HelperIRPara<float>* helper_weights,
    cudaStream_t stream)
{
    stream_ = stream;
    const int nc = num_cam_, seq_len = num_query_, embed_dims = hidden_units_, ch = input_tensors->at(0).shape[1];

    /// OUT
    auto d_output_prj = (float*)output_tensors->at(0).data; /* [L, embed_dims] */

    /// IN
    const float* in_x[]{(const float*)input_tensors->at(0).data,
                        (const float*)input_tensors->at(1).data,
                        (const float*)input_tensors->at(2).data};
    const std::vector<std::vector<size_t>> in_x_shapes{
        input_tensors->at(0).shape, input_tensors->at(1).shape, input_tensors->at(2).shape}; /* [NC, Ch, _, _] */

    auto in_l2i = (const float*)input_tensors->at(3).data; /* [NC,4,4] */

    /// weights
    const auto w_fc_attention_bias = weights->attention_weights.bias;
    const auto w_output_proj_weight = weights->output_proj.kernel;

    fp32::L2IDivImgShpe(d_l2i_norm_,  /* [nc, 4, 4] */
                        in_l2i,       /* from node in; [nc, 4, 4] */
                        d_img_shape_, /* from node in; [1, 2] */
                        nc,
                        4, /* l2i h  */
                        4, /* l2i w  */
                        stream_);

    fp32::L2IxReferencePoints((float*)d_rp_matmuled_,    /* rp after matmul; [1, NC, 4, L] */
                              (const float*)d_l2i_norm_, /* lidar2img, has div img_shapes; [1, NC, 4, 4] */
                              helper_weights->rp_norm,   /* rp after norm; [4, L], const */
                              nc * 4,                    /* l2i_norm h  */
                              4,                         /* l2i_norm w  */
                              seq_len,                   /* rp_norm w */
                              cublas_wrapper_,
                              stream_);

    // FT_SAVE<float>("ir.ca.fs.rfpcat.log", {4, seq_len}, (float*)helper_weights->rp_norm);
    // FT_SAVE<float>("ca.fs.rfpcammatmul.log", {1, nc, 4, seq_len}, (float*)d_rp_matmuled_);

    fp32::ReferencePointsCamAndMask(d_fs_mask_,  /* last out; [1, L, NC, 1] */
                                    d_rpc_norm_, /* rpc after norm;  [1, NC, L, 1, 2] */
                                    (const float*)d_rp_matmuled_,
                                    nc,
                                    4u,
                                    seq_len,
                                    stream_);

    // reference_points_3d, output, mask = feature_sampling_onnx(
    //     value, reference_points, self.pc_range, kwargs['img_shape'], kwargs['lidar2img'])
    fp32::BatchedBilinearGridSample(d_fs_output_, /* last out; [1, Ch, L, NC, 1, 4] */
                                    (const float*)d_rpc_norm_,
                                    in_x,
                                    in_x_shapes,
                                    seq_len,
                                    stream_);
    // FT_SAVE<float>("ca.fs.rfpcam.log", {nc, seq_len, 1, 2}, (float*)d_rpc_norm_);       // passed
    // FT_SAVE<float>("ca.fs_output.log", {ch, seq_len, nc, 1, 4}, (float*)d_fs_output_);  // bad
    // FT_SAVE<uint8_t>("ca.fs_mask.log", {seq_len, nc, 1, 1}, (uint8_t*)d_fs_mask_);      // passed

    //////// helper_weights->attention_weights_output is follow:
    // attention_weights = self.attention_weights(query).view(
    //     bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

    /////// d_reduce_output_ is output of last
    // attention_weights = attention_weights.sigmoid() * mask
    // output = output * attention_weights
    // output = output.sum(-1).sum(-1).sum(-1)
    // output = output.permute(2, 0, 1)
    fp32::TwoMulReduceSum(
        d_reduce_output_, /* output; [L, ch] */
        d_pencoder_bufs_1_,
        helper_weights->attention_weights_output, /* AttentionWeightsFc output without bias, [L, nc*1*4]  */
        w_fc_attention_bias,                      /* AttentionWeightsFc which bias here [1, nc*1*4] */
        (const float*)d_fs_output_,
        (const uint8_t*)d_fs_mask_,
        ch,
        seq_len,
        nc,
        1,
        4,
        stream_);
    // FT_SAVE<float>("ir.ca.attention_weights.out_nobias.log",
    //                {seq_len, nc * 1 * 4},
    //                (float*)helper_weights->attention_weights_output);
    // FT_SAVE<float>("ca.output_permute.log", {seq_len, ch}, (float*)d_reduce_output_);

    // output = self.output_proj(output)
    /// d_output_prj without `w_output_proj_bias`
    fp32::OutputProjFc(d_output_prj, /* last output;  [L, embed_dims] */
                       (const float*)d_reduce_output_,
                       w_output_proj_weight, /* [embed_dims, embed_dims]  */
                       seq_len,
                       embed_dims,
                       embed_dims,
                       cublas_wrapper_,
                       stream_);
    // FT_SAVE<float>("ca.out.subbias.log", {seq_len, embed_dims}, (float*)d_output_prj);
}

}  // namespace fastertransformer
