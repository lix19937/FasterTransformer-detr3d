
#include "cuda_utils.h"
#include "nativeMMWrapper.h"

namespace fastertransformer {
namespace native {

template<bool transA = false, bool transB = false>
__global__ void MatrixMulCUDAV2FF(float* C,
                                  const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  int M,
                                  int K,
                                  int N,
                                  int strideA,
                                  int strideB,
                                  int strideC,
                                  float alpha = 1.f,
                                  float beta = 0.0f)
{
    float Cvalue = 0;
    // M x K   K x N
    int row = blockIdx.x * blockDim.x + threadIdx.x;  ///  m
    int col = blockIdx.y * blockDim.y + threadIdx.y;  ///  n
    int p = blockIdx.z;                               /// batch

    if (row >= M || col >= N) {
        return;
    }

    if (!transA && !transB) {
        // A row，B col
        for (int i = 0; i < K; ++i) {
            Cvalue += __ldg(&A[row * K + i + p * strideA]) * __ldg(&B[i * N + col + p * strideB]);
        }

        C[row * N + col + p * strideC] = alpha * Cvalue + beta;
    }
    else if (!transA && transB) {  //
        for (int i = 0; i < K; ++i) {
            Cvalue += __ldg(&A[row * K + i + p * strideA]) * __ldg(&B[col * K + i + p * strideB]);
        }

        C[col * M + row + p * strideC] = alpha * Cvalue + beta;
    }
    else {
        printf("not support\n");
    }
}

template<bool transA = false, bool transB = false>
__global__ void MatrixMulCUDAV2(half* C,
                                const half* __restrict__ A,
                                const half* __restrict__ B,
                                int M,
                                int K,
                                int N,
                                int strideA,
                                int strideB,
                                int strideC,
                                float alpha = 1.f,
                                float beta = 0.0f)
{
    float Cvalue = 0;
    // MxK   KxN
    int row = blockIdx.x * blockDim.x + threadIdx.x;  ///  m
    int col = blockIdx.y * blockDim.y + threadIdx.y;  ///  n
    int p = blockIdx.z;                               /// batch

    if (row >= M || col >= N) {
        return;
    }

    if (!transA && !transB) {  /// CUBLAS_OP_N,   CUBLAS_OP_N
        // A row，B col
        for (int i = 0; i < K; ++i) {
            Cvalue += __half2float(__hmul(__ldg(&A[row * K + i + p * strideA]), __ldg(&B[i * N + col + p * strideB])));
        }

        C[row * N + col + p * strideC] = alpha * Cvalue + beta;
    }
    else if (!transA && transB) {  /// CUBLAS_OP_T,   CUBLAS_OP_N
        for (int i = 0; i < K; ++i) {
            Cvalue += __half2float(__hmul(__ldg(&A[row * K + i + p * strideA]), __ldg(&B[col * K + i + p * strideB])));
        }

        C[col * M + row + p * strideC] = alpha * Cvalue + beta;
    }
    else {
        printf("not support\n");
    }
}

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
          const float f_alpha,
          const float f_beta,
          const bool is_fp16,
          cudaStream_t stream)
{
    int __n = m, __m = n, __k = k;  // map  to CUBLAS_OP_N, CUBLAS_OP_N

    int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 grid((__m + threads.x - 1) / threads.x, (__n + threads.y - 1) / threads.y);

    if (is_fp16) {
        MatrixMulCUDAV2<<<grid, threads, 0, stream>>>(
            (half*)C, (const half*)B, (const half*)A, __m, __k, __n, 0, 0, 0, f_alpha, f_beta);
    }
    else {
        MatrixMulCUDAV2FF<<<grid, threads, 0, stream>>>(
            (float*)C, (const float*)B, (const float*)A, __m, __k, __n, 0, 0, 0, f_alpha, f_beta);
    }
    sync_check_cuda_error();
}

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
                        const float f_alpha,
                        const float f_beta,
                        const bool is_fp16,
                        cudaStream_t stream)
{
    bool a_trans = transa == cublasOperation_t::CUBLAS_OP_N;
    bool b_trans = transb == cublasOperation_t::CUBLAS_OP_N;

    int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 grid((m + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y, batch_count);

    if (is_fp16) {
        if (a_trans && b_trans) {           /// N  N
            int __n = m, __m = n, __k = k;  /// map  to CUBLAS_OP_N, CUBLAS_OP_N

            int block_size = 32;
            dim3 threads(block_size, block_size);
            dim3 grid((__m + threads.x - 1) / threads.x, (__n + threads.y - 1) / threads.y, batch_count);
            MatrixMulCUDAV2<false, false><<<grid, threads, 0, stream>>>(
                (half*)C, (const half*)B, (const half*)A, __m, __k, __n, strideB, strideA, strideC, f_alpha, f_beta);
        }
        else if (!a_trans && b_trans) {  ///  T  N
            MatrixMulCUDAV2<false, true><<<grid, threads, 0, stream>>>(
                (half*)C, (const half*)A, (const half*)B, m, k, n, strideA, strideB, strideC, f_alpha, f_beta);
        }
        else {
            printf("not support %s %d\n", __FILE__, __LINE__);
        }
    }
    else {
        if (a_trans && b_trans) {           /// N  N
            int __n = m, __m = n, __k = k;  /// map to CUBLAS_OP_N, CUBLAS_OP_N

            int block_size = 32;
            dim3 threads(block_size, block_size);
            dim3 grid((__m + threads.x - 1) / threads.x, (__n + threads.y - 1) / threads.y, batch_count);
            MatrixMulCUDAV2FF<false, false><<<grid, threads, 0, stream>>>(
                (float*)C, (const float*)B, (const float*)A, __m, __k, __n, strideB, strideA, strideC, f_alpha, f_beta);
        }
        else if (!a_trans && b_trans) {  ///  T  N
            MatrixMulCUDAV2FF<false, true><<<grid, threads, 0, stream>>>(
                (float*)C, (const float*)A, (const float*)B, m, k, n, strideA, strideB, strideC, f_alpha, f_beta);
        }
        else {
            printf("not support %s %d\n", __FILE__, __LINE__);
        }
    }
    sync_check_cuda_error();
}

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
                 const int batch_count)
{
    /// NOT USE AT PSERENT
    printf("not enter, not support %s %d\n", __FILE__, __LINE__);
}

}  // namespace native
}  // namespace fastertransformer
