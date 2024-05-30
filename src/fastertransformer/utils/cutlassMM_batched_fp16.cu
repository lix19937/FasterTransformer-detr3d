
#include "cutlassMM.h"
#include <iostream>
#include <map>

#include "cutlass/gemm/device/gemm_batched.h"

namespace fastertransformer {
namespace cutlass_native {

// clang-format off

// n08
// cutlass_tensorop_h16816gemm_128x64_64x3_nn_align8  stridedBatchedGemm  NN {32 512 512, 32 512 32, 16384 262144 16384, 8}   
using GemmBatched_8_32_512_512 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout B
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout C
    float, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm80,            // ArchTag_
    cutlass::gemm::GemmShape<128, 64, 64>, // ThreadblockShape_
    cutlass::gemm::GemmShape<64, 32, 64>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 16>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        8,
        float,
        float>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,//ThreadblockSwizzle_
    3, // Stages
    8, // AlignmentA
    8, // AlignmentB  
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// n09
// cutlass_tensorop_h1688gemm_128x64_32x2_tn_align8  stridedBatchedGemm  TN {512 512 32, 32 32 512, 16384 16384 262144, 8}  
using GemmBatched_8_512_512_32 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::ColumnMajor, // layerout B
    cutlass::half_t, cutlass::layout::ColumnMajor, // layerout C
    float, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm75,            // ArchTag_
    cutlass::gemm::GemmShape<128, 64, 32>, // ThreadblockShape_
    cutlass::gemm::GemmShape<64, 32, 32>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 8>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        8,
        float,
        float>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,//ThreadblockSwizzle_
    2, // Stages
    8, // AlignmentA
    8, // AlignmentB  
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// just compile with cublas (which is m n k order )
const static std::map<std::string, std::vector<int> > cublas_cutlass_table{
    //cublas batch mnk, A_OP,  B_OP|  cutlass [A_layout,  B_layout,  C_layout]
    // {"1_4_4_2",        {'T',   'N',           'R',       'C',       'C'}}, // ndebug
    {"8_32_512_512",   {'N',   'N',           'R',       'R',       'R'}}, // n08
    {"8_512_512_32",   {'T',   'N',           'R',       'C',       'C'}}  // n09 
  };
// clang-format on

#define cutlass_gemm_batched_exec(batch_count,                                                                         \
                                  cbl_m,                                                                               \
                                  cbl_n,                                                                               \
                                  cbl_k,                                                                               \
                                  m,                                                                                   \
                                  n,                                                                                   \
                                  k,                                                                                   \
                                  a,                                                                                   \
                                  b,                                                                                   \
                                  c,                                                                                   \
                                  d,                                                                                   \
                                  lda,                                                                                 \
                                  ldb,                                                                                 \
                                  ldc,                                                                                 \
                                  ldd,                                                                                 \
                                  stride_a,                                                                            \
                                  stride_b,                                                                            \
                                  stride_c,                                                                            \
                                  stride_d,                                                                            \
                                  alpha,                                                                               \
                                  beta,                                                                                \
                                  stream)                                                                              \
    do {                                                                                                               \
        GemmBatched_##batch_count##_##cbl_m##_##cbl_n##_##cbl_k gop;                                                   \
        GemmBatched_##batch_count##_##cbl_m##_##cbl_n##_##cbl_k::Arguments args({m, n, k},                             \
                                                                                {a, lda},                              \
                                                                                stride_a,                              \
                                                                                {b, ldb},                              \
                                                                                stride_b,                              \
                                                                                {c, ldc},                              \
                                                                                stride_c,                              \
                                                                                {d, ldd},                              \
                                                                                stride_d,                              \
                                                                                {alpha, beta},                         \
                                                                                batch_count);                          \
        /* size_t workspace_size = GemmBatched_##N##_##M##_##K::get_workspace_size(args); */                           \
        /* std::cout << "\t>> gop workspace_size：" << workspace_size << std::endl; */                                \
        auto status = gop(args, nullptr, stream);                                                                      \
        if (status != cutlass::Status::kSuccess) {                                                                     \
            std::cout << "\t>> gemm_op error," << int(status) << std::endl;                                            \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

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
    char key[64]{0};
    sprintf(key, "%d_%d_%d_%d", batch_count, m, n, k);
    auto it = cublas_cutlass_table.find(std::string(key));
    if (it != cublas_cutlass_table.end()) {
        std::vector<int> bmnk;
        Split(bmnk, it->first, '_');
        if (bmnk.size() != 4) {
            printf("[cudaacc] bad key:%s,%s:%d\n", it->first.c_str(), __FILE__, __LINE__);
            exit(0);
        }

        // in cutlass view
        int64_t stride_a{0}, stride_b{0};
        cutlass::half_t const* a{nullptr};
        cutlass::half_t const* b{nullptr};
        cutlass::half_t* c = (cutlass::half_t*)C;

        int m{0}, n{0}, k{0};
        if (it->second[0] == 'N' && it->second[1] == 'N') {
            n = bmnk[1], m = bmnk[2], k = bmnk[3];
            a = (cutlass::half_t const*)B;
            b = (cutlass::half_t const*)A;
            stride_a = strideB;
            stride_b = strideA;
        }
        else if (it->second[0] == 'T' && it->second[1] == 'N') {
            m = bmnk[1], n = bmnk[2], k = bmnk[3];
            a = (cutlass::half_t const*)A;
            b = (cutlass::half_t const*)B;
            stride_a = strideA;
            stride_b = strideB;
        }

        int lda = it->second[2] == 'C' ? m : k;
        int ldb = it->second[3] == 'C' ? k : n;
        int ldc = it->second[4] == 'C' ? m : n;

        switch (HASH_STRING_PIECE(key)) {
            case "8_32_512_512"_HASH:
                cutlass_gemm_batched_exec(8,
                                          32,
                                          512,
                                          512,
                                          m,
                                          n,
                                          k,
                                          a,
                                          b,
                                          c,
                                          c,
                                          lda,
                                          ldb,
                                          ldc,
                                          ldc,
                                          stride_a,
                                          stride_b,
                                          strideC,
                                          strideC,
                                          f_alpha,
                                          f_beta,
                                          stream);
                break;

            case "8_512_512_32"_HASH:
                cutlass_gemm_batched_exec(8,
                                          512,
                                          512,
                                          32,
                                          m,
                                          n,
                                          k,
                                          a,
                                          b,
                                          c,
                                          c,
                                          lda,
                                          ldb,
                                          ldc,
                                          ldc,
                                          stride_a,
                                          stride_b,
                                          strideC,
                                          strideC,
                                          f_alpha,
                                          f_beta,
                                          stream);
                break;

            default:
                printf("[cudaacc] bad key:%s,%s:%d\n", it->first.c_str(), __FILE__, __LINE__);
                break;
        }
    }
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
}

}  // namespace cutlass_native
}  // namespace fastertransformer
