/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#pragma once

#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void invokeSVRegUpdate(T* qk_buf,
                                       const T* attn_mask,
                                       const T* relative_pos_bias,
                                       const int batch_size,
                                       const int num_head,
                                       const int window_num,
                                       const int window_len,
                                       float qk_scale,
                                       cudaStream_t stream);

}  // namespace fastertransformer
