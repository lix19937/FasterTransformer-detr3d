/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#pragma once

#include "src/fastertransformer/utils/cuda_utils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace fastertransformer {

namespace sv {

template<typename T>
void invokeAddBias(T* output, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasRelu(T* output, const T* bias, const int m, const int n, cudaStream_t stream);

void invokeAddBias(float* output, const half* in, const half* bias, const int m, const int n, cudaStream_t stream);
void invokeAddBias(float* output, const float* in, const float* bias, const int m, const int n, cudaStream_t stream);

}  // namespace sv

}  // namespace fastertransformer
