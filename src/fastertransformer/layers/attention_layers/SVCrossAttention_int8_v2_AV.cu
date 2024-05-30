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

/// all value[0,1,2] are int8, kc/32hw32, value[3] is maxpool2d in plugin

/// just for AVOD

namespace avt {
namespace fake_int8_v2 {

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

__global__ void ReferencePointsCamAndMaskPermuteKernel(uint8_t* __restrict__ mask,
                                                       half* __restrict__ rpc,
                                                       const half* __restrict__ rpc_matmuled,
                                                       const float* __restrict__ pol_datas,
                                                       const float* __restrict__ cxy_cropxseyse_oxy,
                                                       const int* __restrict__ img_shape,
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

    float reference_points_cam_23 = __ldg(&rpc_matmuled[idx_23]);
    float reference_points_cam_01 = __ldg(&rpc_matmuled[idx_01]);
    float reference_points_cam_12 = __ldg(&rpc_matmuled[idx_12]);

    float eps = 1e-5f;
    bool corners_flag = reference_points_cam_23 > 0;
    bool _mask = reference_points_cam_23 > eps;

    float div = max(reference_points_cam_23, 0.01);
    float m = reference_points_cam_01 / div;
    float n = reference_points_cam_12 / div;
    float r = sqrtf(m * m + n * n);
    float theta = corners_flag ? atanf(r) : 3.14 - atanf(r);

    // [1, NC, 1, L] * [1, 4, 1, 1] (will bc)
    float theta_d = /**/ theta * __ldg(&pol_datas[nc_idx * 5])          /* first line */
                    + theta * theta * __ldg(&pol_datas[nc_idx * 5 + 1]) /* second line */
                    + theta * theta * theta * __ldg(&pol_datas[nc_idx * 5 + 2])
                    + theta * theta * theta * theta * __ldg(&pol_datas[nc_idx * 5 + 3])
                    + theta * theta * theta * theta * theta * __ldg(&pol_datas[nc_idx * 5 + 4]);

    float cdist = r > eps ? theta_d / r : 1;
    // [1, NC, 2, L] *  [1, NC, 1, L]
    m = m * cdist;
    n = n * cdist;

    // center_xy = cxy_cropxseyse_oxy[...,:2]
    // crop_cfg_inputs = cxy_cropxseyse_oxy[...,2:6]  #crop_x_start crop_x_end crop_y_start crop_y_end
    // ori_inputs = cxy_cropxseyse_oxy[...,6:] # [1, 4, 2]

    // #  [1, 4, 1]   -    [1, 4, 512] + [1, 4, 1]
    // reference_points_cam[..., 0,:] = ori_inputs[...,0:1] - (reference_points_cam[..., 0,:] + center_xy[..., 0:1])
    // reference_points_cam[..., 1,:] = ori_inputs[...,1:2] - (reference_points_cam[..., 1,:] + center_xy[..., 1:2])
    m = __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 6]) - (m + __ldg(&cxy_cropxseyse_oxy[nc_idx * 8]));
    n = __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 7]) - (n + __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 1]));

    //                    # [1,4] * [1,4]
    // crop_x_start = crop_cfg_inputs[:, :, 0] * ori_inputs[:, :, 0]
    // crop_x_end   = crop_cfg_inputs[:, :, 1] * ori_inputs[:, :, 0]
    // crop_y_start = crop_cfg_inputs[:, :, 2] * ori_inputs[:, :, 1]
    // crop_y_end   = crop_cfg_inputs[:, :, 3] * ori_inputs[:, :, 1]
    auto crop_x_start = __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 2]) * __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 6]);
    auto crop_x_end = __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 3]) * __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 6]);
    auto crop_y_start = __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 4]) * __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 7]);
    auto crop_y_end = __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 5]) * __ldg(&cxy_cropxseyse_oxy[nc_idx * 8 + 7]);

    // auto scale_x = img_shape[1] / (crop_x_end - crop_x_start);
    // auto scale_y = img_shape[0] / (crop_y_end - crop_y_start);

    //  // [1, 4, 512] -  [1, 4, 1]
    // m = (m - crop_x_start) * scale_x / img_shape[1];
    // n = (n - crop_y_start) * scale_y / img_shape[0];
    //////////////////////////////////////////////////////////////////////

    // [1, 4, 512] -  [1, 4, 1]
    m = (m - crop_x_start) / (crop_x_end - crop_x_start);
    n = (n - crop_y_start) / (crop_y_end - crop_y_start);

    /// from  [1, NC, 2, L] permute to [1, NC, L, 2], rpc restore layout
    unsigned int dim_acc_d[]{0, L * 2, 2, 1};
    auto idx_01_d = nc_idx * dim_acc_d[1] + l_idx * dim_acc_d[2];
    auto idx_12_d = nc_idx * dim_acc_d[1] + l_idx * dim_acc_d[2] + 1;

    /// from [1, NC, L, 1] permute to [1, L, NC, 1], mask restore layout
    unsigned int dim_acc_e[]{0, NC, 1, 1};
    auto idx_01_e = l_idx * dim_acc_e[1] + nc_idx * dim_acc_e[2];

    m = m + m - 1.f;  //(m - 0.5) * 2;
    n = n + n - 1.f;  //(n - 0.5) * 2;
    rpc[idx_01_d] = m;
    rpc[idx_12_d] = n;

    if (_mask) {
        if (fabsf(m) < 1 && fabsf(n) < 1) {
            mask[idx_01_e] = 1;  // true
        }
        else {
            mask[idx_01_e] = 0;  // false
        }
    }
    else {
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

/// NCHW --> NC/32HW32,  idx in linear plane, dst_idx is in NC/32HW32 plane
__forceinline__ __device__ void convert_to_chw32plane(const size_t idx, const size_t area, size_t* dst_idx)
{
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

    const auto voc = area << 5;
    *dst_idx = (idx / voc * voc) + (idx % area << 5) + (idx / area & 31);
}

/// stride:2, 2, kernel_shape:1,1, pads:0
__forceinline__ __device__ void find_maxpool2d_chw32_src(const size_t idx,
                                                         const size_t area,
                                                         const size_t w,
                                                         const int8_t* __restrict__ input,
                                                         const float scale,
                                                         half* __restrict__ value)
{
    auto AREA = area << 2;  /// 4 * area, W = 2 * w, before pool
    auto raw_idx = (idx / area * AREA) + ((idx % area / w) << 1) * (w << 1) + (idx % area % w << 1);

    size_t dst_idx;
    convert_to_chw32plane(raw_idx, AREA, &dst_idx);
    *value = __ldg(&input[dst_idx]) * scale;
}

__forceinline__ __device__ void get_interpolation(int inp_ptr_NC,
                                                  half* __restrict__ sum,
                                                  const int8_t* __restrict__ input,
                                                  const float scale,
                                                  const half* __restrict__ grid,
                                                  const int blockIdx_y,
                                                  const int inp_sH,
                                                  const int inp_sC,
                                                  const int out_sC,
                                                  const int grid_offset,
                                                  const int inp_H,
                                                  const int inp_W)
{
    half2 coord = __ldg((const half2*)&(grid[grid_offset]));
    half2 size = make_half2(__int2half_rn(inp_W), __int2half_rn(inp_H));
    half2 type1P = make_half2(__int2half_rn(1), __int2half_rn(1));

    half2 res1 = __hmul2(__hadd2(coord, type1P), size);
    half2 ixy = __hmul2(__hsub2(res1, type1P), __float2half2_rn(0.5f));

    half2 ixy_nw = h2floor(ixy);
    half2 ixy_nw_1 = __hadd2(ixy_nw, type1P);

    half2 sub12 = __hsub2(ixy_nw_1, ixy);
    half2 sub34 = __hsub2(ixy, ixy_nw);

    half nw = __hmul(sub12.x, sub12.y);
    half ne = __hmul(sub34.x, sub12.y);
    half sw = __hmul(sub12.x, sub34.y);
    half se = __hmul(sub34.x, sub34.y);

    int ix_nw_i = __half2int_rd(ixy_nw.x);
    int iy_nw_i = __half2int_rd(ixy_nw.y);

    int ix_nw_1_i = __half2int_rd(ixy_nw_1.x);
    int iy_nw_1_i = __half2int_rd(ixy_nw_1.y);

    int mul1 = iy_nw_i * inp_sH;    // iy_nw == iy_ne
    int mul3 = iy_nw_1_i * inp_sH;  // iy_sw == iy_se

    int offset1 = mul1 + ix_nw_i;
    int offset2 = mul1 + ix_nw_1_i;
    int offset3 = mul3 + ix_nw_i;
    int offset4 = mul3 + ix_nw_1_i;

    inp_ptr_NC += inp_sC * blockIdx_y;
    const size_t area = inp_H * inp_W;
    size_t dst_idx;
    if (within_bounds_2d(iy_nw_i, ix_nw_i, inp_H, inp_W)) {
        convert_to_chw32plane(inp_ptr_NC + offset1, area, &dst_idx);
        *sum = __hfma(__float2half(__ldg(&input[dst_idx]) * scale), nw, *sum);
    }

    if (within_bounds_2d(iy_nw_i, ix_nw_1_i, inp_H, inp_W)) {
        convert_to_chw32plane(inp_ptr_NC + offset2, area, &dst_idx);
        *sum = __hfma(__float2half(__ldg(&input[dst_idx]) * scale), ne, *sum);
    }

    if (within_bounds_2d(iy_nw_1_i, ix_nw_i, inp_H, inp_W)) {
        convert_to_chw32plane(inp_ptr_NC + offset3, area, &dst_idx);
        *sum = __hfma(__float2half(__ldg(&input[dst_idx]) * scale), sw, *sum);
    }

    if (within_bounds_2d(iy_nw_1_i, ix_nw_1_i, inp_H, inp_W)) {
        convert_to_chw32plane(inp_ptr_NC + offset4, area, &dst_idx);
        *sum = __hfma(__float2half(__ldg(&input[dst_idx]) * scale), se, *sum);
    }
}

__forceinline__ __device__ void get_interpolation_with_maxpool(int inp_ptr_NC,
                                                               half* __restrict__ sum,
                                                               const int8_t* __restrict__ input,
                                                               const float scale,
                                                               const half* __restrict__ grid,
                                                               const int blockIdx_y,
                                                               const int inp_sH,
                                                               const int inp_sC,
                                                               const int out_sC,
                                                               const int grid_offset,
                                                               const int inp_H,
                                                               const int inp_W)
{
    half2 coord = __ldg((const half2*)&(grid[grid_offset]));
    half2 size = make_half2(__int2half_rn(inp_W), __int2half_rn(inp_H));
    half2 type1P = make_half2(__int2half_rn(1), __int2half_rn(1));

    half2 res1 = __hmul2(__hadd2(coord, type1P), size);
    half2 ixy = __hmul2(__hsub2(res1, type1P), __float2half2_rn(0.5f));

    half2 ixy_nw = h2floor(ixy);
    half2 ixy_nw_1 = __hadd2(ixy_nw, type1P);

    half2 sub12 = __hsub2(ixy_nw_1, ixy);
    half2 sub34 = __hsub2(ixy, ixy_nw);

    half nw = __hmul(sub12.x, sub12.y);
    half ne = __hmul(sub34.x, sub12.y);
    half sw = __hmul(sub12.x, sub34.y);
    half se = __hmul(sub34.x, sub34.y);

    int ix_nw_i = __half2int_rd(ixy_nw.x);
    int iy_nw_i = __half2int_rd(ixy_nw.y);

    int ix_nw_1_i = __half2int_rd(ixy_nw_1.x);
    int iy_nw_1_i = __half2int_rd(ixy_nw_1.y);

    int mul1 = iy_nw_i * inp_sH;
    int mul3 = iy_nw_1_i * inp_sH;

    int offset1 = mul1 + ix_nw_i;
    int offset2 = mul1 + ix_nw_1_i;
    int offset3 = mul3 + ix_nw_i;
    int offset4 = mul3 + ix_nw_1_i;

    inp_ptr_NC += inp_sC * blockIdx_y;
    const size_t area = inp_H * inp_W;
    half value;

    if (within_bounds_2d(iy_nw_i, ix_nw_i, inp_H, inp_W)) {
        find_maxpool2d_chw32_src(inp_ptr_NC + offset1, area, inp_W, input, scale, &value);
        *sum = __hfma(value, nw, *sum);
    }

    if (within_bounds_2d(iy_nw_i, ix_nw_1_i, inp_H, inp_W)) {
        find_maxpool2d_chw32_src(inp_ptr_NC + offset2, area, inp_W, input, scale, &value);
        *sum = __hfma(value, ne, *sum);
    }

    if (within_bounds_2d(iy_nw_1_i, ix_nw_i, inp_H, inp_W)) {
        find_maxpool2d_chw32_src(inp_ptr_NC + offset3, area, inp_W, input, scale, &value);
        *sum = __hfma(value, sw, *sum);
    }

    if (within_bounds_2d(iy_nw_1_i, ix_nw_1_i, inp_H, inp_W)) {
        find_maxpool2d_chw32_src(inp_ptr_NC + offset4, area, inp_W, input, scale, &value);
        *sum = __hfma(value, se, *sum);
    }
}

/////////////////////////////////////////////////////////////////////////
const int grid_H = 512, grid_W = 1;

const int inp_C = 256;
const int out_H = 512;
const int inp_sW = 1;

const int inp_H = 72;
const int inp_W = 184;
const int inp_H2 = 36;
const int inp_W2 = 92;
const int inp_H3 = 18;
const int inp_W3 = 46;
const int inp_H4 = 9;
const int inp_W4 = 23;

const int inp_sH = inp_sW * inp_W;
const int inp_sC = inp_sH * inp_H;
const int inp_sN = inp_sC * inp_C;

const int inp_sH2 = inp_W2;
const int inp_sC2 = inp_sH2 * inp_H2;
const int inp_sN2 = inp_sC2 * inp_C;

const int inp_sH3 = inp_W3;
const int inp_sC3 = inp_sH3 * inp_H3;
const int inp_sN3 = inp_sC3 * inp_C;

const int inp_sH4 = inp_W4;
const int inp_sC4 = inp_sH4 * inp_H4;
const int inp_sN4 = inp_sC4 * inp_C;

const int grid_sCoor = 1;
const int grid_sW = grid_sCoor * 2;
const int grid_sH = grid_sW * grid_W;
const int grid_sN = grid_sH * grid_H;

const int out_sW = 1;
const int out_sH = out_sW * grid_W;
const int out_sC = out_sH * out_H;
const int out_sN = out_sC * inp_C;

// const size_t CH = inp_C * out_H;  ///  256 * 512 2^17
// const int ch = 256, L = 512;
/////////////////////////////////////////////////////////////////////////

/// for all size int8
template<int FEATURE_NC>
__global__ void grid_sampler_2d_forward_kernel_n6c256gh512gw1_chw32(
    const int nthreads,
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ input2,
    const int8_t* __restrict__ input3,
    const float scale,
    const float scale2,
    const float scale3,
    half* __restrict__ reduce_output,
    const half* __restrict__ grid /* __restrict__ reference_points_cam after norm */,
    const half* __restrict__ mul_out, /* from */
    const int mul_out_dims_length,
    const int rate,
    const int reduce_num)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nthreads)
        return;

    __shared__ half2 shmm_x[128];  //// lix19937
    __shared__ half2 shmm_y[128];

    const half2* mul_ptr = (const half2*)mul_out;

    // auto w = i % grid_W; // fixed 0
    auto h = i & 511;  // i % grid_H
    auto n = threadIdx.y;

    // auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
    // auto in_offset = n * out_sN + h * out_sH + w * out_sW + out_sC * blockIdx.y;
    auto grid_offset = n * grid_sN + h * grid_sH;
    auto in_offset = n * out_sN + h * out_sH + out_sC * blockIdx.y;

    half2 sum12 = __float2half2_rn(0.f), sum34 = sum12;  //__float2half2_rn(0.f);

    get_interpolation(
        n * inp_sN, &sum12.x, input, scale, grid, blockIdx.y, inp_sH, inp_sC, out_sC, grid_offset, inp_H, inp_W);

    get_interpolation(
        n * inp_sN2, &sum12.y, input2, scale2, grid, blockIdx.y, inp_sH2, inp_sC2, out_sC, grid_offset, inp_H2, inp_W2);

    get_interpolation(
        n * inp_sN3, &sum34.x, input3, scale3, grid, blockIdx.y, inp_sH3, inp_sC3, out_sC, grid_offset, inp_H3, inp_W3);

    get_interpolation_with_maxpool(
        n * inp_sN4, &sum34.y, input3, scale3, grid, blockIdx.y, inp_sH4, inp_sC4, out_sC, grid_offset, inp_H4, inp_W4);

    ///////////////////////////////////////////////////
    // int n_idx = in_offset / CH; // CH = inp_C * out_H
    // int chw_idx = in_offset % CH;
    // int out_offset = (chw_idx * inp_N + n_idx) * stack_sz + stack_idx;
    ///////////////////////////////////////////////////

    // auto n_idx = in_offset >> 17;                                               // in_offset / CH;
    // auto chw_idx = in_offset & 131071;                                          // in_offset % CH;
    auto out_offset = ((in_offset & 131071) * FEATURE_NC + (in_offset >> 17)) << 1;  //(chw_idx * inp_N + n_idx) << 1;

    auto _tid = threadIdx.x * blockDim.y + threadIdx.y;
    shmm_x[_tid] = __hmul2(sum12, __ldg(&mul_ptr[out_offset % mul_out_dims_length]));
    shmm_y[_tid] = __hmul2(sum34, __ldg(&mul_ptr[(out_offset + 1) % mul_out_dims_length]));
    __syncthreads();

    if (threadIdx.y + FEATURE_NC / 2 < blockDim.y) {  ///
        shmm_x[_tid] = __hadd2(shmm_x[_tid], shmm_x[_tid + FEATURE_NC / 2]);
        shmm_y[_tid] = __hadd2(shmm_y[_tid], shmm_y[_tid + FEATURE_NC / 2]);
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        //// FEATURE_NC == 4
        shmm_x[_tid] = __hadd2(shmm_x[_tid], shmm_x[_tid + 1]);
        shmm_y[_tid] = __hadd2(shmm_y[_tid], shmm_y[_tid + 1]);

        half2 ret = __hadd2(shmm_x[_tid], shmm_y[_tid]);

        int bid = blockIdx.x + blockIdx.y * gridDim.x;
        unsigned int out_idx = bid * rate + _tid / reduce_num;
        unsigned int r = out_idx >> 9;                       // out_idx / L;  L = 512
        unsigned int c = out_idx & 511;                      // out_idx % L;
        reduce_output[(c << 8) + r] = __hadd(ret.x, ret.y);  // c * ch;   ch =256
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


// [NC*4, 4] * [4, L] -->  [NC*4, L] or [1, NC, 4, L]  rpc_matmuled = l2i_norm * rp_norm
void L2IxReferencePoints(half* rpc_matmuled,  /* reference_points_cam after rp matmul; [1, NC, 4, L] */
                         const half* l2i,     /* lidar2img [NC*4, 4] */
                         const half* rp_norm, /* rp has norm; [4, L] */
                         int m,               /* nc *4 */
                         int k,               /* 4 */
                         int n,               /* l */
                         cublasMMWrapper* cublas_wrapper,
                         cudaStream_t stream)
{
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, rp_norm, n, l2i, k, rpc_matmuled, n, 1.f, 0.f);
}

// fusion some op
void ReferencePointsCamAndMask(uint8_t* mask,            /* mask, last out; [1, L, NC, 1] */
                               half* rpc_norm,           /* reference_points_cam after norm; [1, NC, L, 2] */
                               const half* rpc_matmuled, /* reference_points_cam after torch.matmul; [1, NC, 4, L]*/
                               const float* pol_datas,
                               const float* cxy_cropxseyse_oxy,
                               const int* img_shape,
                               const unsigned int nc,
                               const unsigned int w, /* 4 */
                               const unsigned int l,
                               cudaStream_t stream)
{
    /// here we make sure L < 1024, and w == 4, Here will be improve
    dim3 grid(nc, 1);
    dim3 block(l, 1);
    ReferencePointsCamAndMaskPermuteKernel<<<grid, block, 0, stream>>>(
        mask, rpc_norm, rpc_matmuled, pol_datas, cxy_cropxseyse_oxy, img_shape, nc, w, l);
}

void BatchGridsampleTwoMulReduceSum(
    half* reduce_output, /* last output */
    half* buf,           /* mul_out */
    const half* attention_weights /* from AttentionWeightsFc output */,
    const half* fc_attention_bias, /* AttentionWeightsFc which bias here  */
    const uint8_t* fs_mask,
    const half* grids,                                       /* reference_points_cam after norm; [1, NC, L, 2] */
    const int8_t* mlvl_feats[],                              /* size:3 */
    const std::vector<std::vector<size_t>>& mlvl_feats_dims, /* size:3, [NC, Ch, _, _] */
    const float scale_list[],                                /* size:3 */
    const int ch,
    const int l,
    const int nc,
    const int m,
    const int k,
    cudaStream_t stream)
{
    int reduce_len = nc * m * k;
    AddBiasAttentionWeightsSigmoidMaskKernel<<<reduce_len, l, 0, stream>>>(
        buf, attention_weights, fc_attention_bias, fs_mask, reduce_len);

    dim3 block(32, nc);  /// lix19937
    dim3 grid((grid_H + block.x - 1) / block.x, inp_C);
    int scale = block.y * block.x / reduce_len * 4;  // each_block can do num `reducesum`

    grid_sampler_2d_forward_kernel_n6c256gh512gw1_chw32<4>
        <<<grid, block, 0, stream>>>(grid_H,
                                     mlvl_feats[0],
                                     mlvl_feats[1],
                                     mlvl_feats[2],
                                     scale_list[0],
                                     scale_list[1],
                                     scale_list[2],
                                     reduce_output,
                                     grids,
                                     buf,                /* mul_out */
                                     l * reduce_len / 2, /* mul_out_dims_length */
                                     scale,
                                     reduce_len / 4);
}

/// [seq_len, embed_dims] * [embed_dims, 24] --> [seq_len, 24] -> [seq_len, 6, 1, 4]
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
}  // namespace fake_int8_v2
}  // namespace avt

template<>
void SVCrossAttentionLayer<half>::__forward_fake_int8_v2_AV_branch(
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
    const int8_t* in_x[]{(const int8_t*)input_tensors->at(2).data,
                         (const int8_t*)input_tensors->at(3).data,
                         (const int8_t*)input_tensors->at(4).data};
    const std::vector<std::vector<size_t>> in_x_shapes{
        input_tensors->at(2).shape, input_tensors->at(3).shape, input_tensors->at(4).shape}; /* [NC, Ch, _, _] */
    const float in_x_scales[]{input_tensors->at(2).scale, input_tensors->at(3).scale, input_tensors->at(4).scale};

    /// IN from net
    auto in_l2i = (const half*)input_tensors->at(5).data;                 /* [1, NC,4,4] */
    auto in_pol_datas = (const float*)input_tensors->at(6).data;          /* [4, 5] */
    auto in_cxy_cropxseyse_oxy = (const float*)input_tensors->at(7).data; /* [4, 8] */

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

    /// in_rp (half) -->  d_rp_norm_ (half)
    avt::fake_int8_v2::ReferencePointsNorm(d_rp_norm_,                /* rp after norm; [1, 4, num_query] */
                                           inner_rp,                     /* rp from node in; [1, num_query, 3] */
                                           (const float*)d_pc_range_, /* from node attribute */
                                           seq_len,
                                           3, /* ReferencePoints width 3  */
                                           stream_);

    /// in_l2i (half) * d_rp_norm_ (half)  -->  d_rp_matmuled_ (half)
    avt::fake_int8_v2::L2IxReferencePoints(d_rp_matmuled_,          /* rp after matmul; [1, NC, 4, L] */
                                           (const half*)in_l2i,     /* lidar2img; [1, 6, 4, 4] */
                                           (const half*)d_rp_norm_, /* rp after norm; [4, L] */
                                           nc * 4,
                                           4,
                                           seq_len,
                                           cublas_wrapper_,
                                           stream_);

    /// d_rp_matmuled_(half) ,pol_datas,cxy_cropxseyse_oxy,d_img_shape_ --> d_fs_mask_(uint8) + d_rpc_norm_ (half)
    avt::fake_int8_v2::ReferencePointsCamAndMask(
        d_fs_mask_,                  /* uint8_t, last out; [1, L, NC, 1] */
        d_rpc_norm_,                 /* half, after norm which as grid sample in;  [1, NC, L, 2] */
        (const half*)d_rp_matmuled_, /* half, [1, NC, 4, L] */
        (const float*)in_pol_datas,
        (const float*)in_cxy_cropxseyse_oxy,
        (const int*)d_img_shape_,
        nc,
        4u,
        seq_len,
        stream_);

    avt::fake_int8_v2::AttentionWeightsFc(d_attention_weights_output_, /* output; [L, 24] , no bias*/
                                          inner_query_embbed,          /* [L, embed_dims] */
                                          w_fc_attention_weight,       /* [embed_dims, 24] */
                                          seq_len,
                                          embed_dims,
                                          nc * 4,
                                          cublas_wrapper_,
                                          stream_);

    avt::fake_int8_v2::BatchGridsampleTwoMulReduceSum(
        d_reduce_output_,                         /* half, reduce_output [L, ch] */
        d_pencoder_bufs_1_,                       /* half, buf space [L, 24] */
        (const half*)d_attention_weights_output_, /* const half, from AttentionWeightsFc output */
        w_fc_attention_bias,        /* const half, fc_attention_bias, AttentionWeightsFc which bias here */
        (const uint8_t*)d_fs_mask_, /* fs_mask */
        (const half*)d_rpc_norm_,   /* grids, reference_points_cam after norm; [1, NC, L, 2] */
        in_x,                       /* mlvl_feats[],  size:3 */
        in_x_shapes,                /* mlvl_feats_dims,  size:3; [NC, Ch, _, _] */
        in_x_scales,                /* scale_list[], size:3 */
        ch,
        seq_len,
        nc,
        1,
        4,
        stream_);

    avt::fake_int8_v2::OutputProjFc(d_output, /* last output;  [L, embed_dims] */
                                    (const half*)d_reduce_output_,
                                    w_output_proj_weight, /* [embed_dims, embed_dims]  */
                                    seq_len,
                                    embed_dims,
                                    embed_dims,
                                    cublas_wrapper_,
                                    stream_);

    avt::fake_int8_v2::PositionEncoder(d_pos_feat,       /* last output;  [L, embed_dims] */
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
void SVCrossAttentionLayer<half>::__forward_magic_fake_int8_v2_AV_branch(
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
    const int8_t* in_x[]{(const int8_t*)input_tensors->at(0).data,
                         (const int8_t*)input_tensors->at(1).data,
                         (const int8_t*)input_tensors->at(2).data};
    const std::vector<std::vector<size_t>> in_x_shapes{
        input_tensors->at(0).shape, input_tensors->at(1).shape, input_tensors->at(2).shape}; /* [NC, Ch, _, _] */

    const float in_x_scales[]{input_tensors->at(0).scale, input_tensors->at(1).scale, input_tensors->at(2).scale};

    /// IN from net
    auto in_l2i = (const half*)input_tensors->at(3).data;                 /* [NC,4,4] */
    auto in_pol_datas = (const float*)input_tensors->at(4).data;          /* [4, 5] */
    auto in_cxy_cropxseyse_oxy = (const float*)input_tensors->at(5).data; /* [4, 8] */

    /// weights
    const auto w_fc_attention_bias = weights->attention_weights.bias;
    const auto w_output_proj_weight = weights->output_proj.kernel;

    /// half --> half
    avt::fake_int8_v2::L2IxReferencePoints((half*)d_rp_matmuled_,   /* rp after matmul; [1, NC, 4, L] */
                                           (const half*)in_l2i,     /* lidar2img; [1, NC, 4, 4] */
                                           helper_weights->rp_norm, /* rp after norm; [4, L], const */
                                           nc * 4,                  /* l2i_norm h  */
                                           4,                       /* l2i_norm w  */
                                           seq_len,                 /* rp_norm w */
                                           cublas_wrapper_,
                                           stream_);

    avt::fake_int8_v2::ReferencePointsCamAndMask(d_fs_mask_,  /* uint8_t, last out; [B, Ch, L, NC, 1, 1] */
                                                 d_rpc_norm_, /* half, rpc after norm;  [1, NC, L, 2] */
                                                 (const half*)d_rp_matmuled_,
                                                 (const float*)in_pol_datas,
                                                 (const float*)in_cxy_cropxseyse_oxy,
                                                 (const int*)d_img_shape_,
                                                 nc,
                                                 4u,
                                                 seq_len,
                                                 stream_);

    avt::fake_int8_v2::BatchGridsampleTwoMulReduceSum(
        d_reduce_output_,                         /* half, reduce_output */
        d_pencoder_bufs_1_,                       /* half, buf */
        helper_weights->attention_weights_output, /* const half, from AttentionWeightsFc output */
        w_fc_attention_bias,        /* const half, fc_attention_bias, AttentionWeightsFc which bias here */
        (const uint8_t*)d_fs_mask_, /* fs_mask */
        (const half*)d_rpc_norm_,   /* grids, reference_points_cam after norm; [1, NC, L, 2] */
        in_x,                       /* mlvl_feats[],  size:3 */
        in_x_shapes,                /* mlvl_feats_dims,  size:3 [4]; [NC, Ch, _, _] */
        in_x_scales,                /* scale_list[], size:3 */
        ch,
        seq_len,
        nc,
        1,
        4,
        stream_);

    avt::fake_int8_v2::OutputProjFc(d_output, /* last output;  [L, embed_dims] */
                                    (const half*)d_reduce_output_,
                                    w_output_proj_weight, /* [embed_dims, embed_dims]  */
                                    seq_len,
                                    embed_dims,
                                    embed_dims,
                                    cublas_wrapper_,
                                    stream_);
}

}  // namespace fastertransformer
