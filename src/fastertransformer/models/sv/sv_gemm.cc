/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#include "src/fastertransformer/utils/gemm_test/sv_gemm_func.h"
#include "src/fastertransformer/utils/gemm_test/gemm_func.h"

//#include "src/fastertransformer/utils/gemm_test/encoder_igemm_func.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft = fastertransformer;

int main(int argc, char* argv[])
{
    if (argc != 8) {
        printf(
            "[ERROR] vit_gemm batch_size seq_len embed_dim head_number with_cls_token data_type int8_mode \n");
        printf("e.g. ./build/bin/sv_gemm 1 512 256 8 1 1 0 \n");
        return 0;
    }

    const int batch_size = atoi(argv[1]);
    int seq_len = atoi(argv[2]);
    const int embed_dim = atoi(argv[3]);
    const int head_num = atoi(argv[4]);
    const int with_cls_token = atoi(argv[5]);
    const ft::CublasDataType data_type = static_cast<ft::CublasDataType>(atoi(argv[6]));  // 0 FP32, 1 FP16, 2 BF 16
    const int int8_mode = atoi(argv[7]);

    printf("[INFO] arguments: \n");
    printf("  batch_size: %d \n", batch_size);
    printf("  seq_len: %d \n", seq_len);
    printf("  embed_dim: %d \n", embed_dim);
    printf("  head_num: %d \n", head_num);
    printf("  with_cls_token: %d \n", with_cls_token);
    printf("  data_type: %d \n", data_type);
    printf("  int8_mode: %d \n", int8_mode);

    if (embed_dim % head_num != 0) {
        printf("[ERROR] Invalid embed_dim and head_num, (e=%d mod h=%d) != 0\n", embed_dim, head_num);
    }

    if (atoi(argv[6]) == 1 && seq_len > 384 && seq_len % 8 != 0) {
        seq_len = (seq_len + 7) / 8 * 8;
    }
    const int size_per_head = embed_dim / head_num;

    std::cout << std::endl;

    void* gemm_test_buf;
    size_t buf_size_in_byte =
        ft::SVcalGemmTestBufSizeInByte(batch_size, seq_len, head_num, size_per_head, 0, 0, int8_mode, data_type);
    size_t total, free;
    ft::check_cuda_error(cudaMemGetInfo(&free, &total));
    if (free < buf_size_in_byte + 10 * 1024 * 1024) {
        printf("[ERROR] There is no enough device memory for gemm test!\n"
               " %ld Bytes is needed, but only %ld Bytes is free.\n",
               buf_size_in_byte,
               free);
        gemm_test_buf = NULL;
        return -1;
    }
    else {
        ft::deviceMalloc(reinterpret_cast<char**>(&gemm_test_buf), buf_size_in_byte, false);
    }

    if (int8_mode != 0) { 
      /// enter int8
       // ft::generate_encoder_igemm_config(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
    else if (data_type == ft::FLOAT_DATATYPE) {
        ft::generate_sv_gemm_config<float>(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
    else if (data_type == ft::HALF_DATATYPE) {
        ft::generate_sv_gemm_config<half>(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
    else {
        printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
        return -1;
    }

    ft::check_cuda_error(cudaFree(gemm_test_buf));
    return 0;
}
