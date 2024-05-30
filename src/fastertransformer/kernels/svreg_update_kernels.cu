/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "src/fastertransformer/kernels/svreg_update_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"

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
                                       cudaStream_t stream)
{
    // const int word_per_thread = 1;
    // dim3 grid(window_len / word_per_thread, window_num * num_head, batch_size);
    // dim3 block((window_len + 31) / 32 * 32);
    // softmax_kernel<<<grid, block, 0, stream>>>(qk_buf,
    //                                            attn_mask,
    //                                            relative_pos_bias,
    //                                            batch_size,
    //                                            num_head,
    //                                            window_num,
    //                                            window_len,
    //                                            window_len * window_len,
    //                                            qk_scale);
}

template void invokeSVRegUpdate(float* qk_buf,
                                                const float* attn_mask,
                                                const float* relative_pos_bias,
                                                const int batch_size,
                                                const int num_head,
                                                const int window_num,
                                                const int window_len,
                                                const float qk_scale,
                                                cudaStream_t stream);

template void invokeSVRegUpdate(half* qk_buf,
                                                const half* attn_mask,
                                                const half* relative_pos_bias,
                                                const int batch_size,
                                                const int num_head,
                                                const int window_num,
                                                const int window_len,
                                                const float qk_scale,
                                                cudaStream_t stream);

}  // namespace fastertransformer
