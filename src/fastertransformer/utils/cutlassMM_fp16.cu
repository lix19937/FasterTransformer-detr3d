
#include "cutlassMM.h"
#include <iostream>
#include <map>

#include "cutlass/gemm/device/gemm.h"

namespace fastertransformer {
namespace cutlass_native {

using __ElementAccumulator = float;

// clang-format off

// nd  for debug 
// cutlass_tensorop_f16_s1688gemm_f16_64x64_32x2_nn_align1   gemm NN {2 4 3, 1.000000 0.000000} 
using GemmNN_1_2_4_3 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // transposed B operand
    cutlass::half_t, cutlass::layout::RowMajor,    // transposed A operand
    cutlass::half_t, cutlass::layout::RowMajor,
    float, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm75,            // ArchTag_
    cutlass::gemm::GemmShape<64, 64, 32>, // ThreadblockShape_
    cutlass::gemm::GemmShape<32, 32, 32>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 8>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        1,
        float,
        float>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//ThreadblockSwizzle_
    2, // Stages
    1, // AlignmentA
    1, // AlignmentB  
    true,// SplitKSerial,
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// n01
// cutlass_tensorop_h16816gemm_64x128_32x6_nn_align8   gemm NN {512 512 256, 512 256 512, 1.000000 0.000000} [line:187]  
using GemmNN_1_512_512_256 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout B
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout C
    __ElementAccumulator, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm80,            // ArchTag_
    cutlass::gemm::GemmShape<64, 128, 32>, // ThreadblockShape_
    cutlass::gemm::GemmShape<32, 64, 32>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 16>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        8,
        __ElementAccumulator,
        __ElementAccumulator>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//ThreadblockSwizzle_
    6, // Stages
    8, // AlignmentA
    8, // AlignmentB  
    true,// SplitKSerial,
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// n02
// cutlass_tensorop_f16_s16816gemm_f16_64x64_32x10_nn_align8   gemm NN {256 512 512, 256 512 256, 1.000000 0.000000} [line:187]  
using GemmNN_1_256_512_512 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout B
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout C
    float, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm80,            // ArchTag_
    cutlass::gemm::GemmShape<64, 64, 32>, // ThreadblockShape_
    cutlass::gemm::GemmShape<32, 32, 32>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 16>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        8,
        float,
        float>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//ThreadblockSwizzle_
    10, // Stages
    8, // AlignmentA
    8, // AlignmentB  
    true,// SplitKSerial,
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// n03
// cutlass_tensorop_h16816gemm_64x64_32x10_nn_align8   gemm NN {256 512 256, 256 256 256, 1.000000 0.000000} [line:187] 
using GemmNN_1_256_512_256 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout B
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout C
    __ElementAccumulator, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm80,            // ArchTag_
    cutlass::gemm::GemmShape<64, 64, 32>, // ThreadblockShape_
    cutlass::gemm::GemmShape<32, 32, 32>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 16>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        8,
        __ElementAccumulator,
        __ElementAccumulator>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//ThreadblockSwizzle_
    10, // Stages
    8, // AlignmentA
    8, // AlignmentB  
    true,// SplitKSerial,
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// n04
// cutlass_tensorop_h16816gemm_64x64_32x10_nn_align4   gemm NN {8   512 256, 8   256 8,   1.000000 0.000000} [line:187] 
using GemmNN_1_8_512_256 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout B
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout C
    __ElementAccumulator, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm80,            // ArchTag_
    cutlass::gemm::GemmShape<64, 64, 32>, // ThreadblockShape_
    cutlass::gemm::GemmShape<32, 32, 32>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 16>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        4,
        __ElementAccumulator,
        __ElementAccumulator>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//ThreadblockSwizzle_
    10, // Stages
    4, // AlignmentA
    4, // AlignmentB  
    true,// SplitKSerial,
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// n05 = n04
// cutlass_tensorop_h16816gemm_64x64_32x10_nn_align4   gemm NN {512 24  4,   512 4 512,   1.000000 0.000000} [line:187] 
using GemmNN_1_512_24_4 = GemmNN_1_8_512_256;

// n06 = n01
// cutlass_tensorop_h16816gemm_64x128_32x6_nn_align8   gemm NN {24  512 256, 24 256 24,   1.000000 0.000000} [line:187] 
using GemmNN_1_24_512_256 = GemmNN_1_512_512_256;

// n07
// cutlass_simt_hgemm_64x128_8x2_nn_align1   gemm NN {256 512 3,   256 3 256,   1.000000 0.000000} [line:187] 
using GemmNN_1_256_512_3 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout B
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout C
    __ElementAccumulator, // ElementAccumulator
    cutlass::arch::OpClassSimt, // OperatorClass_
    cutlass::arch::Sm60,            // ArchTag_
    cutlass::gemm::GemmShape<64, 128, 8>, // ThreadblockShape_
    cutlass::gemm::GemmShape<32, 64, 8>, // WarpShape_
    cutlass::gemm::GemmShape<1, 1, 1>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        1,
        __ElementAccumulator,
        __ElementAccumulator>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//ThreadblockSwizzle_
    2, // Stages
    1, // AlignmentA
    1, // AlignmentB  
    true,// SplitKSerial,
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// n10
// cutlass_tensorop_h1688gemm_64x64_32x2_nn_align1 gemm NN {11   512 256, 11   256 11,   1.000000 0.000000} [line:187] 
using GemmNN_1_11_512_256 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout B
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout C
    __ElementAccumulator, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm75,            // ArchTag_
    cutlass::gemm::GemmShape<64, 64, 32>, // ThreadblockShape_
    cutlass::gemm::GemmShape<32, 32, 32>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 8>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        1,
        __ElementAccumulator,
        __ElementAccumulator>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//ThreadblockSwizzle_
    2, // Stages
    1, // AlignmentA
    1, // AlignmentB  
    true,// SplitKSerial,
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// n11
// cutlass_tensorop_h1688gemm_64x128_32x2_nn_align2 gemm NN {512  16  4,   512 4   512, 1.000000 0.000000} [line:187] 
using GemmNN_1_512_16_4 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout B
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout C
    __ElementAccumulator, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm75,            // ArchTag_
    cutlass::gemm::GemmShape<64, 128, 32>, // ThreadblockShape_
    cutlass::gemm::GemmShape<32, 64, 32>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 8>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        2,
        __ElementAccumulator,
        __ElementAccumulator>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//ThreadblockSwizzle_
    2, // Stages
    2, // AlignmentA
    2, // AlignmentB  
    true,// SplitKSerial,
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// n12
// cutlass_tensorop_h16816gemm_64x64_32x10_nn_align8 gemm NN {16   512 256, 16  256 16,  1.000000 0.000000} [line:187] 
using GemmNN_1_16_512_256 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout A
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout B
    cutlass::half_t, cutlass::layout::RowMajor,    // layerout C
    __ElementAccumulator, // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OperatorClass_
    cutlass::arch::Sm80,            // ArchTag_
    cutlass::gemm::GemmShape<64, 64, 32>, // ThreadblockShape_
    cutlass::gemm::GemmShape<32, 32, 32>, // WarpShape_
    cutlass::gemm::GemmShape<16, 8, 16>,   // InstructionShape_ 
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        8,
        __ElementAccumulator,
        __ElementAccumulator>, // EpilogueOutputOp_
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//ThreadblockSwizzle_
    10, // Stages
    8, // AlignmentA
    8, // AlignmentB  
    true,// SplitKSerial,
    cutlass::arch::OpMultiplyAdd //, Operator_

  >;

// just compatible with cublas (which is m n k order )
const static std::map<std::string, std::vector<int> > cublas_cutlass_table{
  // batch mnk,      | A_OP,  B_OP   cutlass [A_layout,  B_layout,  C_layout]
  // {"1_2_4_3",      {'N',   'N',           'R',       'R',       'R'}}, // nd
  {"1_512_512_256",   {'N',   'N',           'R',       'R',       'R'}}, // n01
  {"1_256_512_512",   {'N',   'N',           'R',       'R',       'R'}}, // n02
  {"1_256_512_256",   {'N',   'N',           'R',       'R',       'R'}}, // n03
  {"1_8_512_256",     {'N',   'N',           'R',       'R',       'R'}}, // n04
  {"1_512_24_4",      {'N',   'N',           'R',       'R',       'R'}}, // n05
  {"1_24_512_256",    {'N',   'N',           'R',       'R',       'R'}}, // n06
  {"1_256_512_3",     {'N',   'N',           'R',       'R',       'R'}}, // n07  
  {"1_11_512_256",    {'N',   'N',           'R',       'R',       'R'}}, // n10
  {"1_512_16_4",      {'N',   'N',           'R',       'R',       'R'}}, // n11
  {"1_16_512_256",    {'N',   'N',           'R',       'R',       'R'}}, // n12
};

// clang-format on

#define cutlass_gemm_exec(                                                                                             \
    batch_count, cbl_m, cbl_n, cbl_k, m, n, k, a, b, c, d, lda, ldb, ldc, ldd, alpha, beta, stream)                    \
    do {                                                                                                               \
        GemmNN_##batch_count##_##cbl_m##_##cbl_n##_##cbl_k gop;                                                        \
        GemmNN_##batch_count##_##cbl_m##_##cbl_n##_##cbl_k::Arguments args(                                            \
            {m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta});                                         \
        /* size_t workspace_size = GemmNN_##N##_##M##_##K::get_workspace_size(args); */                                \
        /* std::cout << "\t>> gop workspace_size：" << workspace_size << std::endl;  */                               \
        auto status = gop(args, nullptr, stream);                                                                      \
        if (status != cutlass::Status::kSuccess) {                                                                     \
            std::cout << "\t>> gemm_op error," << int(status) << std::endl;                                            \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

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
    char key[64]{0};
    sprintf(key, "%d_%d_%d_%d", 1, m, n, k);
    auto it = cublas_cutlass_table.find(std::string(key));
    if (it != cublas_cutlass_table.end()) {
        std::vector<int> bmnk;
        Split(bmnk, it->first, '_');
        if (bmnk.size() != 4) {
            printf("[cudaacc] bad key:%s,%s:%d\n", it->first.c_str(), __FILE__, __LINE__);
            exit(0);
        }

        // in cutlass view
        cutlass::half_t const* a{nullptr};
        cutlass::half_t const* b{nullptr};
        cutlass::half_t* c = (cutlass::half_t*)C;

        int m{0}, n{0}, k{0};
        // here can be use switch
        if (it->second[0] == 'N' && it->second[1] == 'N') {
            n = bmnk[1], m = bmnk[2], k = bmnk[3];
            a = (cutlass::half_t const*)B;
            b = (cutlass::half_t const*)A;
        }
        else if (it->second[0] == 'T' && it->second[1] == 'N') {
            m = bmnk[1], n = bmnk[2], k = bmnk[3];
            a = (cutlass::half_t const*)A;
            b = (cutlass::half_t const*)B;
        }
        else if (it->second[0] == 'T' && it->second[1] == 'T') {
            printf("[cudaacc] TBD %s %s:%d\n", it->first.c_str(), __FILE__, __LINE__);
        }
        else if (it->second[0] == 'N' && it->second[1] == 'T') {
            printf("[cudaacc] TBD %s %s:%d\n", it->first.c_str(), __FILE__, __LINE__);
        }

        int lda = it->second[2] == 'C' ? m : k;
        int ldb = it->second[3] == 'C' ? k : n;
        int ldc = it->second[4] == 'C' ? m : n;

        switch (HASH_STRING_PIECE(key)) {
            case "1_512_512_256"_HASH:
                cutlass_gemm_exec(1, 512, 512, 256, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            case "1_256_512_512"_HASH:
                cutlass_gemm_exec(1, 256, 512, 512, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            case "1_256_512_256"_HASH:
                cutlass_gemm_exec(1, 256, 512, 256, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            case "1_8_512_256"_HASH:
                cutlass_gemm_exec(1, 8, 512, 256, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            case "1_11_512_256"_HASH:
                cutlass_gemm_exec(1, 11, 512, 256, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            case "1_512_24_4"_HASH:
                cutlass_gemm_exec(1, 512, 24, 4, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            case "1_24_512_256"_HASH:
                cutlass_gemm_exec(1, 24, 512, 256, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            case "1_256_512_3"_HASH:
                cutlass_gemm_exec(1, 256, 512, 3, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            case "1_512_16_4"_HASH:
                cutlass_gemm_exec(1, 512, 16, 4, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            case "1_16_512_256"_HASH:
                cutlass_gemm_exec(1, 16, 512, 256, m, n, k, a, b, c, c, lda, ldb, ldc, ldc, f_alpha, f_beta, stream);
                break;

            default:
                printf("[cudaacc] bad key:%s,%s:%d\n", it->first.c_str(), __FILE__, __LINE__);
                break;
        }
    }
}

}  // namespace cutlass_native
}  // namespace fastertransformer
