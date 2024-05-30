/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "src/fastertransformer/kernels/svadd_bias_relu_kernels.h"

namespace fastertransformer {

namespace sv {

///__half22float2
__global__ void invokeAddBiasKernel(float* output, const half* in, const half* bias, const int n)
{
    const int col_index = threadIdx.x;

    auto idx = blockIdx.x * blockDim.x + col_index;
    output[idx] = __hadd(__ldg(&in[idx]), __ldg(&bias[col_index]));
}

void invokeAddBias(float* output, const half* in, const half* bias, const int m, const int n, cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    invokeAddBiasKernel<<<grid, block, 0, stream>>>(output, in, bias, n);
}

__global__ void invokeAddBiasKernel(float* output, const float* in, const float* bias, const int n)
{
    const int col_index = threadIdx.x;

    auto idx = blockIdx.x * blockDim.x + col_index;
    output[idx] = __ldg(&in[idx]) + __ldg(&bias[col_index]);
}

void invokeAddBias(float* output, const float* in, const float* bias, const int m, const int n, cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    invokeAddBiasKernel<<<grid, block, 0, stream>>>(output, in, bias, n);
}

/// for half/float
/// output is [m, n],  bias [n]
template<typename T>
__global__ void invokeAddBiasKernel(T* output, const T* bias, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    // if (col_index < n)
    {
        auto idx = blockIdx.x * blockDim.x + col_index;  /// blockDim.x = n,    blockIdx.x <=m-1
        output[idx] += __ldg(&bias[col_index]);
    }
}

template<>
__global__ void invokeAddBiasKernel(half* output, const half* bias, const int n)
{
    const int col_index = threadIdx.x;  /// blockIdx.y = 0， blockIdx.x <=m-1

    auto idx = blockIdx.x * blockDim.x + col_index;
    half2* output_ptr = (half2*)output;
    const half2* bias_ptr = (const half2*)bias;
    output_ptr[idx] = __hadd2(__ldg(&bias_ptr[col_index]), output_ptr[idx]);
}

template<typename T>
void invokeAddBias(T* output, const T* bias, const int m, const int n, cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);  /// if n < 1024, then blocks_per_row = 1
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    if (std::is_same<T, half>::value && n % 2 == 0) {
        block.x >>= 1;
    }

    invokeAddBiasKernel<<<grid, block, 0, stream>>>(output, bias, n);
}

/// for half will use half2
/// output is [m, n],  bias [n]
template<typename T>
__global__ void invokeAddBiasReluKernel(T* output, const T* bias, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        auto idx = blockIdx.x * blockDim.x + col_index;  /// blockDim.x = n,    blockIdx.x <=m-1
        T with_bias_val = __ldg(&bias[col_index]) + output[idx];
        output[idx] = with_bias_val > T(0.f) ? with_bias_val : T(0.f);
    }
}

template<>
__global__ void invokeAddBiasReluKernel(half* output, const half* bias, const int n)
{
    const int col_index = threadIdx.x;  /// blockIdx.y = 0， blockIdx.x <=m-1
    {
        auto idx = blockIdx.x * blockDim.x + col_index;
        half2* output_ptr = (half2*)output;
        const half2* bias_ptr = (const half2*)bias;
        half2 with_bias_val = __hadd2(__ldg(&bias_ptr[col_index]), output_ptr[idx]);
        const auto type0p = (half)0.0f;
        if (with_bias_val.x < type0p)
            with_bias_val.x = type0p;
        if (with_bias_val.y < type0p)
            with_bias_val.y = type0p;
        output_ptr[idx] = with_bias_val;
    }
}

template<typename T>
void invokeAddBiasRelu(T* output, const T* bias, const int m, const int n, cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);  /// if n < 1024, then blocks_per_row = 1
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    if (std::is_same<T, half>::value) {
        block.x >>= 1;
    }

    invokeAddBiasReluKernel<<<grid, block, 0, stream>>>(output, bias, n);
}

template void invokeAddBias(float* output, const float* bias, const int m, const int n, cudaStream_t stream);

template void invokeAddBias(half* output, const half* bias, const int m, const int n, cudaStream_t stream);

#ifdef ENABLE_BF16
template void
invokeAddBias(__nv_bfloat16* output, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif

template void invokeAddBiasRelu(float* output, const float* bias, const int m, const int n, cudaStream_t stream);

template void invokeAddBiasRelu(half* output, const half* bias, const int m, const int n, cudaStream_t stream);

#ifdef ENABLE_BF16
template void
invokeAddBiasRelu(__nv_bfloat16* output, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif
}  // namespace sv

}  // namespace fastertransformer
