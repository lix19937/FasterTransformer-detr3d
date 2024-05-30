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

/// all value[0,1,2] are fp16, klinear, value[3] is maxpool2d in plugin

namespace half_linear {

__forceinline__ __device__ bool within_bounds_2d(const int h, const int w, const int H, const int W)
{
    return h >= 0 && w >= 0 && h < H && w < W;
}

__global__ void TransposeAndNormKernel(half* __restrict__ out,
                                       const half* __restrict__ in,
                                       const float* __restrict__ range,
                                       const int height,
                                       const int width)
{
    const int BLOCK_DIM = 32;
    __shared__ half block[BLOCK_DIM][BLOCK_DIM + 1];

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
        out[index_out] =
            __half2float(block[threadIdx.x][threadIdx.y]) * __ldg(&range[yIndex + 3]) + __ldg(&range[yIndex]);
    }
    else if (xIndex < height && yIndex == width) {
        unsigned int index_out = yIndex * height + xIndex;
        out[index_out] = 1;
    }
}

__global__ void DivKernel(half* __restrict__ out,
                          const float* __restrict__ in /* net IN  lidar2img */,
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

__global__ void DivKernel_w4(half* __restrict__ out,
                             const float* __restrict__ in /* net IN  lidar2img */,
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
                                                       half* __restrict__ rpc,
                                                       const half* __restrict__ rpc_matmuled,
                                                       const unsigned int NC,
                                                       const unsigned int w,
                                                       const unsigned int L)
{
    auto l_idx = threadIdx.x;  // map to l
    auto nc_idx = blockIdx.x;  // map to nc

    /// for [1, NC, 4, L] of rpc_matmuled   [:,:,2:3,:]
    unsigned int dim_acc[]{4 * L, L, 1};
    auto idx_01 = nc_idx * dim_acc[0] + l_idx;  // 0th row
    auto idx_12 = idx_01 + dim_acc[1];          // 1th row
    auto idx_23 = idx_12 + dim_acc[1];          // 2th row

    float reference_points_cam_23 = __ldg(&rpc_matmuled[idx_23]);
    float reference_points_cam_01 = __ldg(&rpc_matmuled[idx_01]);
    float reference_points_cam_12 = __ldg(&rpc_matmuled[idx_12]);

    /// for [1, NC, L, 2]  rpc
    unsigned int dim_acc_d[]{L * 2, 2, 1};
    auto idx_01_d = nc_idx * dim_acc_d[0] + l_idx * dim_acc_d[1];
    auto idx_12_d = idx_01_d + 1;

    /// from [1, NC, L, 1] permute to [1, L, NC, 1], mask restore layout
    unsigned int dim_acc_e[]{NC, 1, 1};
    auto idx_01_e = l_idx * dim_acc_e[0] + nc_idx * dim_acc_e[1];

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

        if (fabsf(m) < 1 && fabsf(n) < 1) {
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

__forceinline__ __device__ half grid_sampler_compute_source_index(const half coord, const int size)
{
    half size_ = __int2half_rn(size);

    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    half res0 = __hadd(coord, __int2half_rn(1));
    half res1 = __hmul(res0, size);
    half res2 = __hsub(res1, __int2half_rn(1));
    return __hmul(res2, __float2half_rn(0.5));
}

__global__ void
grid_sampler_2d_forward_kernel(const int nthreads,
                               const half* __restrict__ input,
                               const int inp_N,
                               const int inp_C,
                               const int inp_H,
                               const int inp_W,
                               const half* __restrict__ grid,
                               const int grid_H,
                               const int grid_W,
                               half* __restrict__ output /*N = inp_N, C = inp_C, H = grid_H, W = grid_W*/,
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
    int inp_sH = inp_W;  // eq: inp_sW * inp_W
    int inp_sC = inp_sH * inp_H;
    int inp_sN = inp_sC * C;  // eq: inp_sC * inp_C

    int grid_sCoor = 1;
    int grid_sW = 2;                // eq: grid_sCoor * 2
    int grid_sH = out_W << 1;       // eq: grid_sW * grid_W
    int grid_sN = grid_sH * out_H;  // eq: grid_sH * grid_H

    int out_sW = 1;
    int out_sH = out_W;           // eq: out_sW * grid_W
    int out_sC = out_sH * out_H;  // eq: out_sH * grid_H
    int out_sN = out_sC * C;      // eq: out_sC * inp_C

    int CH = C * out_H;

    // #pragma unroll
    // for (; i < nthreads; i += blockDim.x * gridDim.x)
    {
        const int w = i % out_W;
        const int h = (i / out_W) % out_H;
        const int n = i / (out_H * out_W);
        const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

        // get the corresponding input x, y co-ordinates from grid
        half ix0 = __ldg(&grid[grid_offset]);
        half iy0 = __ldg(&grid[grid_offset + grid_sCoor]);
        half ix = grid_sampler_compute_source_index(ix0, inp_W);
        half iy = grid_sampler_compute_source_index(iy0, inp_H);

        // get NE, NW, SE, SW pixel values from (x, y)
        half ix_nw = hfloor(ix);
        half iy_nw = hfloor(iy);

        half one_ = 1;
        half a = __hadd(ix_nw, one_);
        half b = __hadd(iy_nw, one_);

        half ix_ne = a;
        half iy_ne = iy_nw;
        half ix_sw = ix_nw;
        half iy_sw = b;
        half ix_se = a;
        half iy_se = b;

        half sub1 = __hsub(ix_se, ix);
        half sub2 = __hsub(ix, ix_sw);
        half sub3 = __hsub(iy_se, iy);
        half sub4 = __hsub(iy, iy_ne);

        half nw = __hmul(sub1, sub3);
        half ne = __hmul(sub2, sub3);
        half sw = __hmul(sub1, sub4);
        half se = __hmul(sub2, sub4);

        /// calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = n * inp_sN;  //// here can be improved
        auto in_offset = n * out_sN + h * out_sH + w * out_sW;

        int iy_nw_ = __half2int_rd(iy_nw);
        int ix_nw_ = __half2int_rd(ix_nw);
        int iy_sw_ = __half2int_rd(iy_sw);
        int ix_ne_ = __half2int_rd(ix_ne);
        int a_i = ix_nw_ + 1;
        int b_i = iy_nw_ + 1;

        int mul1 = iy_nw_ * inp_sH;  // iy_nw == iy_ne
        int mul2 = ix_nw_ * inp_sW;  // ix_nw == ix_sw
        int mul3 = iy_sw_ * inp_sH;  // iy_sw == iy_se
        int mul4 = ix_ne_ * inp_sW;  // ix_ne == ix_se

        int offset1 = mul1 + mul2;
        int offset2 = mul1 + mul4;
        int offset3 = mul3 + mul2;
        int offset4 = mul3 + mul4;

        half sum, mul_tmp;
#pragma unroll
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, in_offset += out_sC) {
            sum = 0.f;
            if (within_bounds_2d(iy_nw_, ix_nw_, inp_H, inp_W)) {
                mul_tmp = __hmul(__ldg(&input[inp_ptr_NC + offset1]), nw);
                sum = __hadd(sum, mul_tmp);
            }

            if (within_bounds_2d(iy_nw, a_i, inp_H, inp_W)) {
                mul_tmp = __hmul(__ldg(&input[inp_ptr_NC + offset2]), ne);
                sum = __hadd(sum, mul_tmp);
            }

            if (within_bounds_2d(iy_sw_, ix_nw_, inp_H, inp_W)) {
                mul_tmp = __hmul(__ldg(&input[inp_ptr_NC + offset3]), sw);
                sum = __hadd(sum, mul_tmp);
            }

            if (within_bounds_2d(b_i, a_i, inp_H, inp_W)) {
                mul_tmp = __hmul(__ldg(&input[inp_ptr_NC + offset4]), se);
                sum = __hadd(sum, mul_tmp);
            }
            ///             inp_N, inp_C, grid_H, grid_W  [6, 256, L, 1],
            /// so simplify to [nc, 256, L] --> [nc, 256x L]
            {
                // printf("CH %d  stack_sz %d\n", CH, stack_sz); /// 131072   4
                int c_idx = in_offset / CH;
                int hw_idx = in_offset % CH;
                int out_offset = (hw_idx * inp_N + c_idx) * stack_sz + stack_idx;
                output[out_offset] = sum;
            }
        }
    }
}

//// special para      __hfma2 --> a * b +c
__global__ void grid_sampler_2d_forward_kernel_n6c256gh512gw1(
    const int nthreads,
    const half* __restrict__ input,
    const int inp_N,
    const int inp_C,  // 256
    const int inp_H,
    const int inp_W,
    const half* __restrict__ grid /* __restrict__ reference_points_cam after norm */,
    const int grid_H,  // 512
    const int grid_W,  // 1
    half* __restrict__ output /* N = inp_N, C = inp_C, H = grid_H, W = grid_W */,
    const int stack_sz,
    const int stack_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nthreads)
        return;

    int inp_sH = inp_W;
    int inp_sC = inp_sH * inp_H;
    int inp_sN = inp_sC << 8;  // eq: inp_sC * inp_C
    int out_sC = grid_H;       // out_H;  // eq: out_sH * grid_H   gird_H grid_W --> 512

    auto tot_x = blockDim.x * gridDim.x;
#pragma unroll
    for (; i < nthreads; i += tot_x) {
        const int h = i & 511;
        const int n = i >> 9;
        const int grid_offset = (n << 10) + (h << 1);

        /// calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = n * inp_sN;    //// here can be improved
        auto in_offset = (n << 17) + h;  // out_sH = 1

        /// get the corresponding input x, y co-ordinates from grid
        half2 coord = __ldg((const half2*)&(grid[grid_offset]));
        half2 size = make_half2(__int2half_rn(inp_W), __int2half_rn(inp_H));
        half2 type1P = make_half2(__int2half_rn(1), __int2half_rn(1));

        half2 res1 = __hmul2(__hadd2(coord, type1P), size);
        half2 ixy = __hmul2(__hsub2(res1, type1P), __float2half2_rn(0.5f));

        half2 ixy_nw = h2floor(ixy);
        half2 ab = __hadd2(ixy_nw, type1P);

        half2 sub13 = __hsub2(ab, ixy);
        half2 sub24 = __hsub2(ixy, ixy_nw);

        half2 sub12 = make_half2(sub13.x, sub24.x);
        half2 sub33 = make_half2(sub13.y, sub13.y);
        half2 sub44 = make_half2(sub24.y, sub24.y);

        half2 mul1233 = __hmul2(sub12, sub33);
        half2 mul1244 = __hmul2(sub12, sub44);

        half nw = mul1233.x;  // 12 33
        half ne = mul1233.y;
        half sw = mul1244.x;  // 12 44
        half se = mul1244.y;
        half iy_nw = ixy_nw.y;

        int ix_nw_ = __half2int_rd(ixy_nw.x);
        int iy_nw_ = __half2int_rd(ixy_nw.y);

        int ix_ne_ = __half2int_rd(ab.x);
        int iy_sw_ = __half2int_rd(ab.y);

        int a_i = ix_nw_ + 1;
        int b_i = iy_nw_ + 1;

        int mul1 = iy_nw_ * inp_sH;
        int mul3 = iy_sw_ * inp_sH;

        inp_ptr_NC += inp_sC * blockIdx.y;
        in_offset += out_sC * blockIdx.y;

        half sum = 0.f;
        half mul_tmp;

        if (within_bounds_2d(iy_nw_, ix_nw_, inp_H, inp_W)) {
            mul_tmp = __hmul(__ldg(&input[inp_ptr_NC + mul1 + ix_nw_]), nw);
            sum = __hadd(sum, mul_tmp);
        }

        if (within_bounds_2d(iy_nw, a_i, inp_H, inp_W)) {
            mul_tmp = __hmul(__ldg(&input[inp_ptr_NC + mul1 + ix_ne_]), ne);
            sum = __hadd(sum, mul_tmp);
        }

        if (within_bounds_2d(iy_sw_, ix_nw_, inp_H, inp_W)) {
            mul_tmp = __hmul(__ldg(&input[inp_ptr_NC + mul3 + ix_nw_]), sw);
            sum = __hadd(sum, mul_tmp);
        }

        if (within_bounds_2d(b_i, a_i, inp_H, inp_W)) {
            mul_tmp = __hmul(__ldg(&input[inp_ptr_NC + mul3 + ix_ne_]), se);
            sum = __hadd(sum, mul_tmp);
        }
        /// inp_N, inp_C, grid_H, grid_W  [6, 256, L, 1],

        int out_offset = (((in_offset & (131071)) * inp_N + (in_offset >> 17)) << 2) + stack_idx;
        output[out_offset] = sum;
    }
}

/// NCHW --> NC/32HW32,  idx in linear plane, dst_idx is in NC/32HW32 plane
// __forceinline__ __device__ void convert_to_chw32plane(const size_t idx, const size_t area, size_t* dst_idx)
// {
//////////////////////////////// python snippet ////////////////////////////
// area = H * W
// voc = 32*area
// # idx is linear plane,
// voc_idx = idx // voc
// row_idx = idx % area
// col_idx = idx // area % 32
// chw32_idx = voc_idx * voc + row_idx*32 + col_idx  # find idx in NCHW32 plane
// dst[idx] = src[chw32_idx]
////////////////////////////////////////////////////////////////////////////
//     const auto voc = area << 5;
//     *dst_idx = (idx / voc * voc) + (idx % area << 5) + (idx / area & 31);
// }

/// stride:2, 2, kernel_shape:1,1, pads:0
__forceinline__ __device__ void find_maxpool2d_src(
    const size_t idx, const size_t area, const size_t w, const half* __restrict__ input, half* __restrict__ value)
{
    auto AREA = area << 2;  /// 4 * area, W = 2 * w, before pool
    auto raw_idx = (idx / area * AREA) + ((idx % area / w) << 1) * (w << 1) + (idx % area % w << 1);

    *value = __ldg(&input[raw_idx]);
}

__global__ void grid_sampler_2d_forward_kernel_n6c256gh512gw1_maxpool(
    const int nthreads,
    const half* __restrict__ input,
    const int inp_N,
    const int inp_C,  // 256
    const int inp_H,
    const int inp_W,
    const half* __restrict__ grid /* __restrict__ reference_points_cam after norm */,
    const int grid_H,  // 512
    const int grid_W,  // 1
    half* __restrict__ output /* N = inp_N, C = inp_C, H = grid_H, W = grid_W*/,
    const int stack_sz,
    const int stack_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nthreads)
        return;

    int inp_sH = inp_W;
    int inp_sC = inp_sH * inp_H;
    int inp_sN = inp_sC << 8;  // eq: inp_sC * inp_C
    int out_sC = grid_H;       // out_H;  // eq: out_sH * grid_H   gird_H grid_W --> 512

    const size_t area = inp_H * inp_W;
    half value;

    auto tot_x = blockDim.x * gridDim.x;
#pragma unroll
    for (; i < nthreads; i += tot_x) {
        const int h = i & 511;
        const int n = i >> 9;
        const int grid_offset = (n << 10) + (h << 1);

        /// calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = n * inp_sN;    //// here can be improved
        auto in_offset = (n << 17) + h;  // out_sH = 1

        /// get the corresponding input x, y co-ordinates from grid
        half2 coord = __ldg((const half2*)&(grid[grid_offset]));
        half2 size = make_half2(__int2half_rn(inp_W), __int2half_rn(inp_H));
        half2 type1P = make_half2(__int2half_rn(1), __int2half_rn(1));

        half2 res1 = __hmul2(__hadd2(coord, type1P), size);
        half2 ixy = __hmul2(__hsub2(res1, type1P), __float2half2_rn(0.5f));

        half2 ixy_nw = h2floor(ixy);
        half2 ab = __hadd2(ixy_nw, type1P);

        half2 sub13 = __hsub2(ab, ixy);
        half2 sub24 = __hsub2(ixy, ixy_nw);

        half2 sub12 = make_half2(sub13.x, sub24.x);
        half2 sub33 = make_half2(sub13.y, sub13.y);
        half2 sub44 = make_half2(sub24.y, sub24.y);

        half2 mul1233 = __hmul2(sub12, sub33);
        half2 mul1244 = __hmul2(sub12, sub44);

        half nw = mul1233.x;  // 12 33
        half ne = mul1233.y;
        half sw = mul1244.x;  // 12 44
        half se = mul1244.y;
        half iy_nw = ixy_nw.y;

        int ix_nw_ = __half2int_rd(ixy_nw.x);
        int iy_nw_ = __half2int_rd(ixy_nw.y);

        int ix_ne_ = __half2int_rd(ab.x);
        int iy_sw_ = __half2int_rd(ab.y);

        int a_i = ix_nw_ + 1;
        int b_i = iy_nw_ + 1;

        int mul1 = iy_nw_ * inp_sH;
        int mul3 = iy_sw_ * inp_sH;

        inp_ptr_NC += inp_sC * blockIdx.y;
        in_offset += out_sC * blockIdx.y;

        half sum = 0.f;
        half mul_tmp;
        if (within_bounds_2d(iy_nw_, ix_nw_, inp_H, inp_W)) {
            find_maxpool2d_src(inp_ptr_NC + mul1 + ix_nw_, area, inp_W, input, &value);

            mul_tmp = __hmul(value, nw);
            sum = __hadd(sum, mul_tmp);
        }

        if (within_bounds_2d(iy_nw, a_i, inp_H, inp_W)) {
            find_maxpool2d_src(inp_ptr_NC + mul1 + ix_ne_, area, inp_W, input, &value);

            mul_tmp = __hmul(value, ne);
            sum = __hadd(sum, mul_tmp);
        }

        if (within_bounds_2d(iy_sw_, ix_nw_, inp_H, inp_W)) {
            find_maxpool2d_src(inp_ptr_NC + mul3 + ix_nw_, area, inp_W, input, &value);

            mul_tmp = __hmul(value, sw);
            sum = __hadd(sum, mul_tmp);
        }

        if (within_bounds_2d(b_i, a_i, inp_H, inp_W)) {
            find_maxpool2d_src(inp_ptr_NC + mul3 + ix_ne_, area, inp_W, input, &value);

            mul_tmp = __hmul(value, se);
            sum = __hadd(sum, mul_tmp);
        }

        int out_offset = (((in_offset & (131071)) * inp_N + (in_offset >> 17)) << 2) + stack_idx;
        output[out_offset] = sum;
    }
}

__global__ void AddBiasAttentionWeightsSigmoidMaskKernel(half* __restrict__ out,
                                                         const half* __restrict__ fc_attention_out,
                                                         const half* __restrict__ fc_attention_bias,
                                                         const uint8_t* __restrict__ fs_mask,
                                                         const int bias_len)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;  // nc *1*4, l
    auto fs_mask_idx = idx >> 2;                       // idx / 4  !!!    [l, nc]
    auto bias_idx = idx % bias_len;                    // bias_len [1, nc*1*4]

    if (__ldg(&fs_mask[fs_mask_idx]) == 0) {
        out[idx] = 0.f;
    }
    else {
        float x = __hadd(__ldg(&fc_attention_out[idx]), __ldg(&fc_attention_bias[bias_idx]));
        out[idx] = 1.f / (1.f + __expf(-x));
    }
}

__global__ void MulAndReducesumKernel(half* __restrict__ reduce_output,
                                      const half* __restrict__ fs_output,
                                      const half* __restrict__ fc_attention_s_m_out,
                                      const int fc_attention_s_m_out_len,
                                      const int scale,
                                      const int reduce_num)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto m_idx = idx % fc_attention_s_m_out_len;
    auto tid = threadIdx.x;
    extern __shared__ half2 shmm[];

    const half2* fs_ptr = (half2*)fs_output;
    const half2* mul_ptr = (half2*)fc_attention_s_m_out;

    half2 d1 = __ldg(&fs_ptr[idx]);
    half2 d2 = __ldg(&mul_ptr[m_idx]);
    shmm[tid] = __hmul2(d1, d2);
    __syncthreads();

#pragma unroll
    for (unsigned int s = reduce_num >> 1; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int offset = 0; offset < blockDim.x; offset += reduce_num) {
                shmm[tid + offset] = __hadd2(shmm[tid + offset], shmm[tid + s + offset]);
            }
        }
        __syncthreads();
    }

    auto bid = blockIdx.x * scale;
#pragma unroll
    for (unsigned int i = 0; i < blockDim.x; i += reduce_num) {
        if (tid == i) {
            /// transpose, [ch,  L] --> [L, ch]
            unsigned int out_idx = bid + i / reduce_num;
            unsigned int r = out_idx >> 9;  /// out_idx / 512
            unsigned int c = out_idx & 511;
            auto sum = __hadd2(shmm[tid], shmm[tid + 2]);
            auto ptr = (half*)&sum;
            reduce_output[(c << 8) + r] = __hadd(ptr[0], ptr[1]);
        }
    }
}

__global__ void IsigmoidAddBiasKernel(half* __restrict__ out, const half* __restrict__ rp /* reference_points [L,1,3]*/)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    const float eps = 1.e-5;

    float x = __ldg(&rp[idx]);
    x = min(max(x, 0.f), 1.f);
    auto x1 = max(x, eps);
    auto x2 = max(1.f - x, eps);
    out[idx] = log(x1 / x2);
}

void ReferencePointsNorm(half* rp_norm,         /* reference_points after norm [L, 3](memory is L*4)  -->  [4, L] */
                         const half* rp,        /* reference_points`s shape B=1, from IN */
                         const float* pc_range, /* from Attri len=6 */
                         const int in_h,        /* L */
                         const int in_w,        /* 3 */
                         cudaStream_t stream)
{
    const int block_w = 32, block_h = 32;
    dim3 grid((in_w + block_w - 1) / block_w, (in_h + block_h - 1) / block_h);
    dim3 block(block_w, block_h);
    TransposeAndNormKernel<<<grid, block, 0, stream>>>(rp_norm, rp, pc_range, in_h, in_w);
}

void L2IDivImgShpe(half* l2i_norm,   /* lidar2img/img_shapes; [ NC, 4, 4] */
                   const float* l2i, /* lidar2img  ; [NC*4, 4], from IN */
                   const int* shape, /* img_shapes ; [1, 4] { B[1], B[0], 1, 1 }, use 2, from IN */
                   const int nc,     /* num of feats */
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
void L2IxReferencePoints(half* rpc_matmuled,   /* reference_points_cam after rp matmul; [1, NC, 4, L] */
                         const half* l2i_norm, /* lidar2img norm, which has div img_shapes; [NC*4, 4] */
                         const half* rp_norm,  /* rp has norm; [4, L] */
                         int m,                /* nc *4 */
                         int k,                /* 4 */
                         int n,                /* l */
                         cublasMMWrapper* cublas_wrapper,
                         cudaStream_t stream)
{
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, rp_norm, n, l2i_norm, k, rpc_matmuled, n, 1.f, 0.f);
}

void ReferencePointsCamAndMask(uint8_t* mask,            /* mask, last out; [1, L, NC, 1] */
                               half* rpc_norm,           /* reference_points_cam after norm; [1, NC, L, 2] */
                               const half* rpc_matmuled, /* reference_points_cam after torch.matmul; [1, NC, 4, L]*/
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

void BatchedBilinearGridSample(half* sampled_feats,      /* sampled_feats, last out; [ CH, L, NC, 1, 4] */
                               const half* rpc_norm,     /* reference_points_cam after norm; [1, NC, L, 2] */
                               const half* mlvl_feats[], /* size:3 */
                               const std::vector<std::vector<size_t>>& mlvl_feats_dims, /* [4]; [ NC, Ch, _, _] */
                               const int seq_len,                                       /* num_cam, num_query */
                               cudaStream_t stream)
{
    const int inp_N = mlvl_feats_dims[0][0];
    const int inp_C = mlvl_feats_dims[0][1];

    const int grid_H = seq_len;
    const int grid_W = 1;              /// here we make sure grid_W == 1 !!!
    const int count = inp_N * grid_H;  /// inp_N * grid_H * grid_W  6 *512
    const int stack_sz = mlvl_feats_dims.size();

    dim3 block(512 / 2, 1);
    dim3 grid((count + block.x - 1) / block.x, inp_C);

    for (int i = 0; i < stack_sz; ++i) {
        const int inp_H = mlvl_feats_dims[i][2];
        const int inp_W = mlvl_feats_dims[i][3];
        grid_sampler_2d_forward_kernel_n6c256gh512gw1<<<grid, block, 0, stream>>>(count,
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

// [seq_len, embed_dims] * [embed_dims, 24] --> [seq_len, 24] -> [seq_len, 6, 1, 4]
void AttentionWeightsFc(half* attention_weights,
                        const half* query_embbed, /* attention_weights = self.attention_weights(query) */
                        const half* fc_attention_weights,
                        const int m,
                        const int k,
                        const int n,
                        cublasMMWrapper* cublas_wrapper,
                        cudaStream_t stream)
{
    cublas_wrapper->Gemm(
        CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, fc_attention_weights, n, query_embbed, k, attention_weights, n, 1.f, 0.f);
}

void TwoMulReduceSum(half* reduce_output, /* last output */
                     half* buf,           /* tmp memory for AddBiasAttentionWeightsSigmoidMask */
                     const half* attention_weights /* from AttentionWeightsFc output */,
                     const half* fc_attention_bias, /* AttentionWeightsFc which bias here  */
                     const half* fs_output,
                     const uint8_t* fs_mask,
                     const int ch,
                     const int l,
                     const int nc,
                     const int m,
                     const int k,
                     cudaStream_t stream)
{
    {
        dim3 block(l);
        dim3 grid(nc * m * k);

        AddBiasAttentionWeightsSigmoidMaskKernel<<<grid, block, 0, stream>>>(
            buf, attention_weights, fc_attention_bias, fs_mask, grid.x);
    }

    {
        const int scale = 4;
        auto block_w = nc * m * k;

        dim3 block(block_w * scale / 2);
        dim3 grid(ch * l / scale);
        MulAndReducesumKernel<<<grid, block, block.x * sizeof(half2), stream>>>(
            reduce_output, fs_output, buf, l * block_w / 2, scale, block_w / 2);
    }
}

void OutputProjFc(half* output,
                  const half* reduce_output,
                  const half* output_proj_weight,
                  const int m,
                  const int k,
                  const int n,
                  cublasMMWrapper* cublas_wrapper,
                  cudaStream_t stream)
{
    cublas_wrapper->Gemm(
        CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, output_proj_weight, n, reduce_output, k, output, n, 1.f, 0.f);
}

void PositionEncoder(half* pos_feat, /* [seq_len, embed_dims] */
                     half* buf[],
                     const half* rp, /* [seq_len, 3] */
                     const half* weights[],
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
}  // namespace half_linear

template<>
void SVCrossAttentionLayer<half>::__forward_half_linear_branch(
    std::vector<fastertransformer::Tensor>* output_tensors,
    const std::vector<fastertransformer::Tensor>* input_tensors,
    const CAttentionWeight<half>* weights)
{
    const int nc = num_cam_, seq_len = num_query_, embed_dims = hidden_units_, ch = input_tensors->at(2).shape[1];

    /// OUT
    auto d_output = (half*)output_tensors->at(0).data;   /* [L, embed_dims] */
    auto d_pos_feat = (half*)output_tensors->at(1).data; /* [L, embed_dims] */

    /// INNER
    auto inner_query_embbed = (const half*)input_tensors->at(0).data; /* [L, embed_dims]  has added with query_pos */
    auto inner_rp = (const half*)input_tensors->at(1).data;           /* [1, L, 3] reference_points */

    /// IN from backbone
    const half* in_x[]{(const half*)input_tensors->at(2).data,
                       (const half*)input_tensors->at(3).data,
                       (const half*)input_tensors->at(4).data};
    const std::vector<std::vector<size_t>> in_x_shapes{
        input_tensors->at(2).shape, input_tensors->at(3).shape, input_tensors->at(4).shape}; /* [NC, Ch, _, _] */

    /// IN from net
    auto in_l2i = (const float*)input_tensors->at(5).data; /* [1, NC,4,4] */

    /// weights
    const auto w_fc_attention_weight = weights->attention_weights.kernel;
    const auto w_fc_attention_bias = weights->attention_weights.bias;
    const auto w_output_proj_weight = weights->output_proj.kernel;
    const half* w_pencoder_weights[]{weights->position_encoder_fc1.kernel,
                                     weights->position_encoder_fc1.bias,
                                     weights->position_encoder_ln1.kernel,
                                     weights->position_encoder_ln1.bias,
                                     weights->position_encoder_fc2.kernel,
                                     weights->position_encoder_fc2.bias,
                                     weights->position_encoder_ln2.kernel,
                                     weights->position_encoder_ln2.bias};

    half_linear::ReferencePointsNorm(d_rp_norm_,                /* rp after norm; [1, 4, num_query] */
                                     inner_rp,                  /* rp from node in; [1, num_query, 3] */
                                     (const float*)d_pc_range_, /* from node attribute */
                                     seq_len,
                                     3, /* ReferencePoints width 3  */
                                     stream_);

    half_linear::L2IDivImgShpe(d_l2i_norm_,  /* [NC, 4, 4] */
                               in_l2i,       /* from node in; [NC, 4, 4] */
                               d_img_shape_, /* from node in; [1, 2], const int */
                               nc,
                               4,
                               4,
                               stream_);

    half_linear::L2IxReferencePoints((half*)d_rp_matmuled_,    /* rp after matmul; [1, NC, 4, L] */
                                     (const half*)d_l2i_norm_, /* lidar2img div img_shapes; [1, 6, 4, 4] */
                                     (const half*)d_rp_norm_,  /* rp after norm; [4, L] */
                                     nc * 4,
                                     4,
                                     seq_len,
                                     cublas_wrapper_,
                                     stream_);

    half_linear::ReferencePointsCamAndMask(d_fs_mask_,  /* last out; [B, Ch, L, NC, 1, 1] */
                                           d_rpc_norm_, /* rpc after norm;  [1, NC, L, 2] */
                                           (const half*)d_rp_matmuled_,
                                           nc,
                                           4u,
                                           seq_len,
                                           stream_);

    half_linear::BatchedBilinearGridSample(d_fs_output_, /* last out; [1, Ch, L, NC, 1, 4] */
                                           (const half*)d_rpc_norm_,
                                           in_x,
                                           in_x_shapes,
                                           seq_len,
                                           stream_);

    half_linear::AttentionWeightsFc(d_attention_weights_output_, /* output; [L, 24] , no bias*/
                                    inner_query_embbed,          /* [L, embed_dims] */
                                    w_fc_attention_weight,       /* [embed_dims, 24] */
                                    seq_len,
                                    embed_dims,
                                    nc * 4,
                                    cublas_wrapper_,
                                    stream_);

    half_linear::TwoMulReduceSum(d_reduce_output_,                         /* output; [L, ch] */
                                 d_pencoder_bufs_1_,                       /* tmp space [L, 24] for current api*/
                                 (const half*)d_attention_weights_output_, /* from AttentionWeightsFc output */
                                 w_fc_attention_bias, /* AttentionWeightsFc which bias here [1, 24] */
                                 (const half*)d_fs_output_,
                                 (const uint8_t*)d_fs_mask_,
                                 ch,
                                 seq_len,
                                 nc,
                                 1,
                                 4,
                                 stream_);

    half_linear::OutputProjFc(d_output, /* last output;  [L, embed_dims] */
                              (const half*)d_reduce_output_,
                              w_output_proj_weight, /* [embed_dims, embed_dims]  */
                              seq_len,
                              embed_dims,
                              embed_dims,
                              cublas_wrapper_,
                              stream_);

    half_linear::PositionEncoder(d_pos_feat,       /* last output;  [L, embed_dims] */
                                 d_pencoder_bufs_, /* tmp space, just for current api */
                                 inner_rp,         /* const input  */
                                 w_pencoder_weights,
                                 seq_len,
                                 3,
                                 embed_dims,
                                 cublas_wrapper_,
                                 stream_);
}

template<>
void SVCrossAttentionLayer<half>::__forward_magic_half_linear_branch(
    std::vector<fastertransformer::Tensor>* output_tensors,
    const std::vector<fastertransformer::Tensor>* input_tensors,
    const CAttentionWeight<half>* weights,
    const HelperIRPara<half>* helper_weights,
    cudaStream_t stream)
{
    stream_ = stream;
    const int nc = num_cam_, seq_len = num_query_, embed_dims = hidden_units_, ch = input_tensors->at(0).shape[1];

    /// OUT
    auto d_output = (half*)output_tensors->at(0).data; /* [L, embed_dims] */

    /// IN from backbone
    const half* in_x[]{(const half*)input_tensors->at(0).data,
                       (const half*)input_tensors->at(1).data,
                       (const half*)input_tensors->at(2).data};
    const std::vector<std::vector<size_t>> in_x_shapes{
        input_tensors->at(0).shape, input_tensors->at(1).shape, input_tensors->at(2).shape}; /* [NC, Ch, _, _] */

    /// IN from net
    auto in_l2i = (const float*)input_tensors->at(3).data; /* [NC,4,4] */

    /// weights
    const auto w_fc_attention_bias = weights->attention_weights.bias;
    const auto w_output_proj_weight = weights->output_proj.kernel;

    half_linear::L2IDivImgShpe(d_l2i_norm_,  /* [nc, 4, 4] */
                               in_l2i,       /* from node in; [nc, 4, 4] */
                               d_img_shape_, /* from node in; [1, 2] */
                               nc,
                               4,
                               4,
                               stream_);

    half_linear::L2IxReferencePoints((half*)d_rp_matmuled_,    /* rp after matmul; [1, NC, 4, L] */
                                     (const half*)d_l2i_norm_, /* lidar2img div img_shapes; [1, 6, 4, 4] */
                                     helper_weights->rp_norm,  /* rp after norm; [4, L], const */
                                     nc * 4,
                                     4,
                                     seq_len,
                                     cublas_wrapper_,
                                     stream_);

    half_linear::ReferencePointsCamAndMask(d_fs_mask_,  /* last out; [B, Ch, L, NC, 1, 1] */
                                           d_rpc_norm_, /* rpc after norm;  [1, NC, L, 2] */
                                           (const half*)d_rp_matmuled_,
                                           nc,
                                           4u,
                                           seq_len,
                                           stream_);

    half_linear::BatchedBilinearGridSample(d_fs_output_, /* last out; [1, Ch, L, NC, 1, 4] */
                                           (const half*)d_rpc_norm_,
                                           in_x,
                                           in_x_shapes,
                                           seq_len,
                                           stream_);

    half_linear::TwoMulReduceSum(d_reduce_output_, /* output; [L, ch] */
                                 d_pencoder_bufs_1_,
                                 helper_weights->attention_weights_output, /* from AttentionWeightsFc output */
                                 w_fc_attention_bias, /* AttentionWeightsFc which bias here [1, 24] */
                                 (const half*)d_fs_output_,
                                 (const uint8_t*)d_fs_mask_,
                                 ch,
                                 seq_len,
                                 nc,
                                 1,
                                 4,
                                 stream_);

    half_linear::OutputProjFc(d_output, /* last output;  [L, embed_dims] */
                              (const half*)d_reduce_output_,
                              w_output_proj_weight, /* [embed_dims, embed_dims]  */
                              seq_len,
                              embed_dims,
                              embed_dims,
                              cublas_wrapper_,
                              stream_);
}

}  // namespace fastertransformer
