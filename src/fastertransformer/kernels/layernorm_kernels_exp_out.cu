/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/kernels/bfloat16_fallback_kenrels.cuh"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

namespace svexp_pos{

// add one output with original output + query_pos 

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt(T* normed_output,
                                                   T* normed_output_with_pos,        
                                                   T* output,
                                                   const T* __restrict bias,
                                                   const T* __restrict residual,
                                                   const T* __restrict query_pos,
                                                   const T* __restrict gamma,
                                                   const T* __restrict beta,
                                                   int m,
                                                   int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    T local_sum = float2type2<T>(0.0f);
#pragma unroll
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = float2type2<T>(0.0f);

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
        s_mean = mean / n * 0.5f;
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
        s_variance = rsqrtf(variance / (n <<1) + 1e-5f);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
        normed_output_with_pos[index] = val + ldg(&query_pos[index]);
    }
}

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(T* normed_output,        
                                                    T* normed_output_with_pos,
                                                    T* output,
                                                    const T* __restrict bias,
                                                    const T* __restrict residual,
                                                    const T* __restrict query_pos,               
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
            tmp = ldg(&output[index]);
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
        s_mean = sums[0] / n / 2;
        s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + 1e-5f);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
        normed_output_with_pos[index] = val + ldg(&query_pos[index]);
    }
}



template<typename T>
__global__ void generalAddBiasResidualLayerNorm(const T* __restrict input,
                                                const T* __restrict query_pos,
                                                const T* __restrict gamma,
                                                const T* __restrict beta,
                                                const T* __restrict bias,
                                                T* output,
                                                T* norm_output,
                                                T* normed_output_with_pos,
                                                int m,
                                                int n)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        auto it = blockIdx.x * n + i; 
        float local_out = (float)(ldg(&input[it]));
        local_out += (float)(output[it]);
        //if (bias != nullptr) 
        {
            local_out += (float)(ldg(&bias[i]));
        }
        output[it] = (T)local_out;
        local_sum += local_out;
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(output[blockIdx.x * n + i]) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-5f);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        //float beta_val = (beta == nullptr) ? 0.0f : (float)(ldg(&beta[i]));
        norm_output[blockIdx.x * n + i] =
            (T)((((float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + (float)(ldg(&beta[i])));
        normed_output_with_pos[blockIdx.x * n + i] = norm_output[blockIdx.x * n + i] + ldg(&query_pos[blockIdx.x * n + i]);
    }
}

#define HALF_LAYERNORM_OPT(UNROLL_FACTOR)                                                                              \
    generalAddBiasResidualLayerNormOpt<T2, true, true, true, true, UNROLL_FACTOR>                                      \
        <<<grid, block, 0, stream>>>((T2*)norm_output,                                                                 \
                                     (T2*)normed_output_with_pos,                                                      \
                                     (T2*)output,                                                                      \
                                     (const T2*)bias,                                                                  \
                                     (const T2*)input,                                                                 \
                                     (const T2*)query_pos,                                                       \
                                     (const T2*)gamma,                                                                 \
                                     (const T2*)beta,                                                                  \
                                     m,                                                                                \
                                     half_n);

#define HALF_LAYERNORM_OPT2(UNROLL_FACTOR)                                                                             \
    generalAddBiasResidualLayerNormOpt2<T2, true, true, true, true, UNROLL_FACTOR>                                     \
        <<<grid, block, 0, stream>>>((T2*)norm_output,                                                                 \
                                     (T2*)normed_output_with_pos,  \
                                     (T2*)output,                                                                      \
                                     (const T2*)bias,                                                                  \
                                     (const T2*)input,                                                                 \
                                     (const T2*)query_pos,                                                       \
                                     (const T2*)gamma,                                                                 \
                                     (const T2*)beta,                                                                  \
                                     m,                                                                                \
                                     half_n);

template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T* output,
                                              T* norm_output,
                                              T* normed_output_with_pos,  
                                              const T* input,
                                              const T* query_pos,
                                              const T* gamma,
                                              const T* beta,
                                              const T* bias,
                                              int m,
                                              int n,
                                              cudaStream_t stream,
                                              int opt_version)
{
    if (opt_version > 0 && sizeof(T) == 2 && n % 2 == 0) {
        dim3 grid(m);
        int half_n = n >>1;
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

        dim3 grid(m);
        dim3 block(min(n, 1024));

        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
        */

        if (n % 32 != 0) {
            block.x = 1024;
        }

        block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

        /* should pay attention to the rsqrt precision*/
        generalAddBiasResidualLayerNorm<T>
            <<<grid, block, 0, stream>>>(input, query_pos, gamma, beta, bias, output, norm_output, normed_output_with_pos, m, n);  // For gpt-3
    }
}

#undef HALF_LAYERNORM_OPT
#undef HALF_LAYERNORM_OPT2

template void invokeGeneralAddBiasResidualPreLayerNorm(float* output,
                                                       float* norm_output,
                                                       float* normed_output_with_pos,
                                                       const float* input,
                                                       const float* query_pos,
                                                       const float* gamma,
                                                       const float* beta,
                                                       const float* bias,
                                                       int m,
                                                       int n,
                                                       cudaStream_t stream,
                                                       int opt_version);

template void invokeGeneralAddBiasResidualPreLayerNorm(half* output,
                                                       half* norm_output,
                                                       half* normed_output_with_pos,
                                                       const half* input,
                                                       const half* query_pos,
                                                       const half* gamma,
                                                       const half* beta,
                                                       const half* bias,
                                                       int m,
                                                       int n,
                                                       cudaStream_t stream,
                                                       int opt_version);

#ifdef ENABLE_BF16
template void invokeGeneralAddBiasResidualPreLayerNorm(__nv_bfloat16* output,
                                                       __nv_bfloat16* norm_output,
                                                       __nv_bfloat16* normed_output_with_pos,
                                                       const __nv_bfloat16* input,
                                                       const __nv_bfloat16* query_pos,
                                                       const __nv_bfloat16* gamma,
                                                       const __nv_bfloat16* beta,
                                                       const __nv_bfloat16* bias,
                                                       int m,
                                                       int n,
                                                       cudaStream_t stream,
                                                       int opt_version);
#endif

}

}  // namespace fastertransformer