
#include "src/fastertransformer/kernels/bfloat16_fallback_kenrels.cuh"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

namespace svpost_relu {

#define sv_eps (1e-5)

// just make output with relu judge

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt(T* normed_output,
                                                   T* output,
                                                   const T* __restrict bias,
                                                   const T* __restrict residual,
                                                   const T* __restrict gamma,
                                                   const T* __restrict beta,
                                                   int m,
                                                   int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    using T1 = typename TypeConverter<T>::Type;  /// half2 --> half

    T type0P = float2type2<T>(.0f);

    T local_sum = float2type2<T>(0.0f);
#pragma unroll
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = type0P;

        if (IS_BIAS) {
            val = hadd2(val, ldg(&bias[i]));
        }
        if (IS_RESIDUAL) {
            val = hadd2(val, ldg(&residual[index]));
        }

        if (IS_OUTPUT) {
            val = hadd2(val, output[index]);
        }
        output[index] = val;
        local_sum = hadd2(local_sum, val);
    }

    mean = blockReduceSum((float)(local_sum.x + local_sum.y));

    if (threadIdx.x == 0) {
        s_mean = mean / (n << 1);
    }
    __syncthreads();

    float local_var_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val = output[blockIdx.x * n + i];
        float diff_1 = (float)(val.x) - s_mean;
        float diff_2 = (float)(val.y) - s_mean;
        local_var_sum += (diff_1 * diff_1 + diff_2 * diff_2);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (n << 1) + 1e-5f);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);
    T1 t1zero = (T1)0.0f;
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }

        if (val.x < t1zero)
            val.x = t1zero;
        if (val.y < t1zero)
            val.y = t1zero;
        normed_output[index] = val;
    }
}

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(T* normed_output,
                                                    T* output,
                                                    const T* __restrict bias,
                                                    const T* __restrict residual,
                                                    const T* __restrict gamma,
                                                    const T* __restrict beta,
                                                    int m,
                                                    int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float x_sum = 0.0f;
    float x2_sum = 0.0f;
    const int b_offset = blockIdx.x * n;
    using T1 = typename TypeConverter<T>::Type;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        float val_1 = 0.0f;
        float val_2 = 0.0f;
        T tmp;

        if (IS_BIAS) {
            tmp = ldg(&bias[i]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        if (IS_RESIDUAL) {
            tmp = ldg(&residual[index]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }

        if (IS_OUTPUT) {
            tmp = output[index];

            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        tmp.x = float2type<T1>(val_1);
        tmp.y = float2type<T1>(val_2);
        output[index] = tmp;
        x_sum += val_1 + val_2;
        x2_sum += val_1 * val_1 + val_2 * val_2;
    }
    float sums[2];
    sums[0] = x_sum;
    sums[1] = x2_sum;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean = sums[0] / (n << 1);
        s_variance = rsqrtf(sums[1] / (n << 1) - s_mean * s_mean + 1e-5f);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);
    T1 t1zero = T1(0.0f);
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }

        if (val.x < t1zero)
            val.x = t1zero;
        if (val.y < t1zero)
            val.y = t1zero;
        normed_output[index] = val;
    }
}

/*******************  invokeAddBiasLayernorm  ***********************/

template<typename T>
__global__ void add_bias_layernorm(T* out, const T* bias, const T* gamma, const T* beta, int n)
{
    int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    int idx = bid * n + tid;

    float local_out = (tid < n) ? (float)(out[idx] + ldg(&bias[tid])) : 0.0f;
    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float diff = (tid < n) ? (local_out - s_mean) : 0.0f;
    variance = blockReduceSum<float>(diff * diff);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-5f;
    }
    __syncthreads();

    if (tid < n) {
        T val = (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(ldg(&gamma[tid])) + (float)(ldg(&beta[tid])));
        out[idx] = val > (T)0.0f ? val : (T)0.0f;
    }
}

template<typename T>
__global__ void
add_bias_layernorm_v2(T* out, const T* __restrict bias, const T* __restrict gamma, const T* __restrict beta, int n)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int offset = bid * n;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; ++i) {
        int col_id = i * blockDim.x + tid;
        local_out[i] = (col_id < n) ? (float)(out[offset + col_id] + ldg(&bias[col_id])) : 0.0f;
        sum += local_out[i];
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        float diff = (col_id < n) ? (local_out[i] - s_mean) : 0.0f;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + 1e-5f);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; ++i) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            auto val =
                (T)((local_out[i] - s_mean) * s_variance * (float)ldg(&gamma[col_id]) + (float)ldg(&beta[col_id]));
            out[offset + col_id] = val > (T)0.0f ? val : (T)0.0f;
        }
    }
}

#define HALF_LAYERNORM_OPT(UNROLL_FACTOR)                                                                              \
    generalAddBiasResidualLayerNormOpt<T2, false, true, true, true, UNROLL_FACTOR><<<grid, block, 0, stream>>>(        \
        (T2*)out, (T2*)out, (const T2*)bias, (const T2*)out, (const T2*)gamma, (const T2*)beta, m, half_n);

#define HALF_LAYERNORM_OPT2(UNROLL_FACTOR)                                                                             \
    generalAddBiasResidualLayerNormOpt2<T2, false, true, true, true, UNROLL_FACTOR><<<grid, block, 0, stream>>>(       \
        (T2*)out, (T2*)out, (const T2*)bias, (const T2*)out, (const T2*)gamma, (const T2*)beta, m, half_n);

template<typename T>
void invokeAddBiasLayernorm(
    T* out, const T* bias, const T* gamma, const T* beta, int m, int n, cudaStream_t stream, int opt_version)
{
    dim3 grid(m);
    if (n % 2 == 0 && std::is_same<T, half>::value && opt_version > 0) {
        int half_n = n / 2;
        int half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int rolls_per_thread = half_n / block.x;
        int unroll_factor = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;
        if (opt_version == 1) {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT(8);
            }
        }
        else {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT2(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT2(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT2(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT2(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT2(8);
            }
        }
    }
    else {
        /// fp32 oth
        int blockSize = (n + 31) / 32 * 32;
        if (blockSize >= 768) {
            blockSize = ((blockSize / 4) + 31) / 32 * 32;
            add_bias_layernorm_v2<T><<<grid, blockSize, 0, stream>>>(out, bias, gamma, beta, n);
        }
        else {
            add_bias_layernorm<T><<<grid, blockSize, 0, stream>>>(out, bias, gamma, beta, n);
        }
    }
}

#undef HALF_LAYERNORM_OPT
#undef HALF_LAYERNORM_OPT2

template void invokeAddBiasLayernorm<float>(float* out,
                                            const float* bias,
                                            const float* gamma,
                                            const float* beta,
                                            int m,
                                            int n,
                                            cudaStream_t stream,
                                            int opt_version);

template void invokeAddBiasLayernorm<half>(half* out,
                                           const half* bias,
                                           const half* gamma,
                                           const half* beta,
                                           int m,
                                           int n,
                                           cudaStream_t stream,
                                           int opt_version);
#ifdef ENABLE_BF16
template void invokeAddBiasLayernorm<__nv_bfloat16>(__nv_bfloat16* out,
                                                    const __nv_bfloat16* bias,
                                                    const __nv_bfloat16* gamma,
                                                    const __nv_bfloat16* beta,
                                                    int m,
                                                    int n,
                                                    cudaStream_t stream,
                                                    int opt_version);
#endif
}  // namespace svpost_relu

}  // namespace fastertransformer
