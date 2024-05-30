
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

#pragma once

namespace fastertransformer {
namespace cutlass_native {

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

inline void Split(std::vector<int>& ret, const std::string& str, char delim = ' ', bool ignore_empty = true)
{
    if (str.empty()) {
        return;
    }

    size_t n = str.size();
    size_t s{0};
    while (s <= n) {
        size_t i = str.find_first_of(delim, s);
        size_t len = 0;
        len = i == std::string::npos ? n - s : i - s;

        if (!ignore_empty || 0 != len) {
            auto tmp = str.substr(s, len);
            ret.push_back(std::move(atoi(tmp.c_str())));
        }

        s += len + 1;
    }
}

// https://stackoverflow.com/questions/2111667/compile-time-string-hashing
// https://cs.chromium.org/chromium/src/base/strings/string_piece.h
constexpr inline size_t HASH_STRING_PIECE(const char* string_piece)
{
    std::size_t result{0};
    for (auto ptr = string_piece; *ptr != 0; ++ptr) {
        result = (result * 131) + *ptr;
    }
    return result;
}

constexpr inline size_t operator""_HASH(const char* string_pice, size_t)
{
    return HASH_STRING_PIECE(string_pice);
}

}  // namespace cutlass_native
}  // namespace fastertransformer
