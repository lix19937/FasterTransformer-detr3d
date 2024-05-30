
#include <cublas_v2.h>
#include <cuda_runtime.h>

#pragma once

namespace fastertransformer {
namespace native {

void Gemm(cublasOperation_t transa,
          cublasOperation_t transb,
          const int m,
          const int n,
          const int k,
          const void* A,
          const int lda,
          const void* B,
          const int ldb,
          void* C,
          const int ldc,
          const float f_alpha = 1.0f,
          const float f_beta = 0.f,
          const bool is_fp16 = true,
          cudaStream_t stream = nullptr);

void stridedBatchedGemm(cublasOperation_t transa,
                        cublasOperation_t transb,
                        const int m,
                        const int n,
                        const int k,
                        const void* A,
                        const int lda,
                        const int64_t strideA,
                        const void* B,
                        const int ldb,
                        const int64_t strideB,
                        void* C,
                        const int ldc,
                        const int64_t strideC,
                        const int batch_count,
                        const float f_alpha = 1.0f,
                        const float f_beta = 0.f,
                        const bool is_fp16 = true,
                        cudaStream_t stream = nullptr);

void batchedGemm(cublasOperation_t transa,
                 cublasOperation_t transb,
                 const int m,
                 const int n,
                 const int k,
                 const void* const* A,
                 const int lda,
                 const void* const* B,
                 const int ldb,
                 void* const* C,
                 const int ldc,
                 const int batch_count);

}  // namespace native
}  // namespace fastertransformer
