
#include "src/fastertransformer/utils/gemm_test/sv_gemm_func.h"

namespace fastertransformer {

template<typename T>
void generate_sv_gemm_config(
    int batch_size, int seq_len, int head_num, int size_per_head, void* buffer_in, bool isAppend)
{
    void* cublas_workspace;
    void* buffer;
    int workSpaceSize;

#ifdef ENABLE_BF16
    if (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) {
#else
    if (std::is_same<T, half>::value) {
#endif  // ENABLE_BF16
        // cublas_workspace_ should be the start pointer of cudaMalloc()
        // to ensure 16B alignemnet
        cublas_workspace = buffer_in;
        buffer = (void*)((char*)cublas_workspace + CUBLAS_WORKSPACE_SIZE);
        workSpaceSize = CUBLAS_WORKSPACE_SIZE;
    }
    else {
        cublas_workspace = nullptr;
        buffer = buffer_in;
        workSpaceSize = 0;
    }

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    // check config
    FILE* fd;
    int line_count = 0;
    if (!isAppend) {
        fd = fopen(GEMM_CONFIG, "w+");
    }
    else {
        fd = fopen(GEMM_CONFIG, "a+");
        std::vector<std::string> config;
        char line[1024];
        while (fgets(line, 1024, fd) != NULL) {
            config.push_back(std::string(line));
        }
        line_count = config.size();
        if (config.size() >= (MAX_CONFIG_NUM * GEMM_NUM + 1))  // 6 cublas/cublasLt, first row is not included
        {
            int startIdx = config.size() - ((MAX_CONFIG_NUM - 1) * GEMM_NUM);
            fclose(fd);
            fd = fopen(GEMM_CONFIG, "w+");
            fprintf(fd, "%s", config[0].c_str());
            for (uint i = startIdx; i < config.size(); i++) {
                fprintf(fd, "%s", config[i].c_str());
            }
            line_count = config.size() - (GEMM_NUM + 3);
        }
    }
    //                        m   n   k
    // algo_map_.size: 0,  1_512_256_256_1
    // algo_map_.size: 0,  1_48_512_4_1
    // algo_map_.size: 0,  1_512_48_256_1
    // algo_map_.size: 0,  1_512_256_3_1
    // algo_map_.size: 0,  1_512_512_256_1
    // algo_map_.size: 0,  1_512_256_512_1
    // algo_map_.size: 0,  1_512_8_256_1
    // algo_map_.size: 0,  8_512_512_32_1
    // algo_map_.size: 0,  8_512_32_512_1

    const int gemm_num = 10;
    int M[gemm_num];
    int N[gemm_num];
    int K[gemm_num];
    int batchCount[gemm_num];
    for (int i = 0; i < gemm_num; ++i)
        batchCount[i] = 1;
    char mess[gemm_num][256];
    float exec_times[gemm_num];

    M[0] = batch_size * seq_len;
    N[0] = head_num * size_per_head;  // hidden_units_
    K[0] = N[0];
    strcpy(mess[0], "1_512_256_256_1");

    M[1] = 48;  //  NC*4
    N[1] = batch_size * seq_len;
    K[1] = 4;
    strcpy(mess[1], "1_48_512_4_1");

    M[2] = batch_size * seq_len;
    N[2] = 48;
    K[2] = head_num * size_per_head;
    strcpy(mess[2], "1_512_48_256_1");

    M[3] = batch_size * seq_len;
    N[3] = head_num * size_per_head;
    K[3] = 3;
    strcpy(mess[3], "1_512_256_3_1");

    M[4] = batch_size * seq_len;
    N[4] = M[4];
    K[4] = head_num * size_per_head;
    strcpy(mess[4], "1_512_512_256_1");

    M[5] = batch_size * seq_len;
    N[5] = head_num * size_per_head;
    K[5] = M[5];
    strcpy(mess[5], "1_512_256_512_1");

    M[6] = batch_size * seq_len;
    N[6] = 8;
    K[6] = head_num * size_per_head;
    strcpy(mess[6], "1_512_8_256_1");

    M[7] = batch_size * seq_len;
    N[7] = 5;
    K[7] = head_num * size_per_head;
    strcpy(mess[7], "1_512_5_256_1");

    //////////////////////////////////////////////////
    M[8] = seq_len;
    N[8] = seq_len;
    K[8] = size_per_head;
    batchCount[8] = batch_size * head_num;
    strcpy(mess[8], "attention batched Gemm1 8_512_512_32_1");

    M[9] = seq_len;
    N[9] = size_per_head;
    K[9] = seq_len;
    batchCount[9] = batch_size * head_num;
    strcpy(mess[9], "attention batched Gemm2 8_512_32_512_1");

    //////////////////////////////////////////////////////////////////////////

    cublasHandle_t cublas_handle;
    check_cuda_error(cublasCreate(&cublas_handle));
    cublasLtHandle_t ltHandle;
    check_cuda_error(cublasLtCreate(&ltHandle));

    cudaDataType_t AType;
    cudaDataType_t BType;
    cudaDataType_t CType;
    cudaDataType_t computeType;
    int startAlgo, endAlgo;
    const int ites = 100;
    struct timeval start, end;

    CublasDataType data_type;
    if (std::is_same<T, float>::value) {
        data_type = FLOAT_DATATYPE;
        AType = CUDA_R_32F;
        BType = CUDA_R_32F;
        CType = CUDA_R_32F;
        computeType = CUDA_R_32F;
        startAlgo = (int)CUBLAS_GEMM_DEFAULT;
        endAlgo = (int)CUBLAS_GEMM_ALGO23;
    }
    else if (std::is_same<T, half>::value) {
        data_type = HALF_DATATYPE;
        AType = CUDA_R_16F;
        BType = CUDA_R_16F;
        CType = CUDA_R_16F;
        computeType = CUDA_R_32F;
        startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        data_type = BFLOAT16_DATATYPE;
        AType = CUDA_R_16BF;
        BType = CUDA_R_16BF;
        CType = CUDA_R_16BF;
        computeType = CUDA_R_32F;
        startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
    }
#endif
    using scaleT = typename ScaleTypeConverter<T, false>::Type;

    scaleT alpha = (scaleT)1.0f;
    scaleT beta = (scaleT)0.0f;

    printf("***Encoder Gemm Testing Begin***\n");
    printf("***Cublas Gemm Testing Begin***\n");
    if (line_count == 0) {
        fprintf(fd,
                "batch_size, seq_len, head_num, size_per_head dataType ### batchCount, n, m, k, algoId, "
                "customOption, tile, numSplitsK, swizzle, reductionScheme, workspaceSize, stages, exec_time\n");
    }
    for (int i = 0; i < gemm_num; ++i) {
        int m = M[i], n = N[i], k = K[i];
        printf("\n-----------------------------\n");
        printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
        T* d_A = (T*)buffer;
        T* d_B = d_A + m * k * batchCount[i];
        T* d_C = d_B + k * n * batchCount[i];

        float exec_time = 99999.0f;
        int fast_algo = 0;
        for (int algo = startAlgo; algo <= endAlgo; algo++) {
            cublasStatus_t status;
            cudaDeviceSynchronize();
            gettimeofday(&start, NULL);
            for (int ite = 0; ite < ites; ++ite) {
                if (i < 8) {
                    status = cublasGemmEx(cublas_handle,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          n,
                                          m,
                                          k,
                                          &alpha,
                                          d_B,
                                          BType,
                                          n,
                                          d_A,
                                          AType,
                                          k,
                                          &beta,
                                          d_C,
                                          CType,
                                          n,
                                          computeType,
                                          static_cast<cublasGemmAlgo_t>(algo));
                }
                else if (i == 8) {
                    status = cublasGemmStridedBatchedEx(cublas_handle,
                                                        CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        seq_len,        //
                                                        seq_len,        //
                                                        size_per_head,  // k
                                                        &alpha,
                                                        d_B,
                                                        BType,
                                                        size_per_head,
                                                        seq_len * size_per_head,
                                                        d_A,
                                                        AType,
                                                        size_per_head,
                                                        seq_len * size_per_head,
                                                        &beta,
                                                        d_C,
                                                        CType,
                                                        seq_len,                /// ldc
                                                        seq_len * seq_len,      /// strideC
                                                        batch_size * head_num,  /// batch_count
                                                        computeType,
                                                        static_cast<cublasGemmAlgo_t>(algo));
                }
                else if (i == 9) {
                    status = cublasGemmStridedBatchedEx(cublas_handle,
                                                        CUBLAS_OP_N,
                                                        CUBLAS_OP_N,
                                                        size_per_head,  // n
                                                        seq_len,        // m
                                                        seq_len,        // k
                                                        &alpha,
                                                        d_B,
                                                        BType,
                                                        size_per_head,
                                                        seq_len * size_per_head,
                                                        d_A,
                                                        AType,
                                                        seq_len,
                                                        seq_len * seq_len,
                                                        &beta,
                                                        d_C,
                                                        CType,
                                                        size_per_head,            /// ldc
                                                        seq_len * size_per_head,  /// strideC
                                                        batch_size * head_num,    /// batch_count
                                                        computeType,
                                                        static_cast<cublasGemmAlgo_t>(algo));
                }

                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("error %d @%s:%d \n", status, __FILE__, __LINE__);
                    break;
                }
            }
            cudaDeviceSynchronize();
            gettimeofday(&end, NULL);
            if (status == CUBLAS_STATUS_SUCCESS) {
                printf("algo_%d costs %.3fms \n", algo, diffTime(start, end) / ites);
                if (diffTime(start, end) / ites < exec_time) {
                    exec_time = diffTime(start, end) / ites;
                    fast_algo = algo;
                }
            }
        }
        printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);

        // for fp16 and bf16, we compare cublasLt
        if (i < 8 && data_type != FLOAT_DATATYPE) {
            printf("***cublasLt Gemm Testing Beign***\n");
            // Let try a fixed number of combinations
            int ALGO_COMBINATIONS = 5000;
            customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
            LtHgemmCustomFind<T, scaleT>(ltHandle,
                                         batch_size,
                                         seq_len,
                                         head_num,
                                         size_per_head,
                                         n,
                                         m,
                                         k,
                                         &alpha,
                                         d_B,
                                         d_A,
                                         &beta,
                                         d_C,
                                         cublas_workspace,
                                         workSpaceSize,
                                         fd,
                                         perfResults,
                                         ALGO_COMBINATIONS);
            if (perfResults[0].time < exec_time) {
                printPerfStructure(
                    batch_size, seq_len, head_num, size_per_head, n, m, k, perfResults[0], fd, data_type, 0);
                exec_time = perfResults[0].time;
            }
            else {
                fprintf(fd,
                        "%d %d %d %d %d ### %d %d %d %d %d -1 -1 -1 -1 -1 -1 -1 %f\n",
                        batch_size,
                        seq_len,
                        head_num,
                        size_per_head,
                        data_type,
                        batchCount[i],
                        n,
                        m,
                        k,
                        fast_algo,
                        exec_time);
            }
            printf("***cublasLt Gemm Testing End***\n");
        }
        else {
            fprintf(fd,
                    "%d %d %d %d %d ### %d %d %d %d %d -1 -1 -1 -1 -1 -1 -1 %f\n",
                    batch_size,
                    seq_len,
                    head_num,
                    size_per_head,
                    data_type,
                    batchCount[i],
                    n,
                    m,
                    k,
                    fast_algo,
                    exec_time);
        }
        exec_times[i] = exec_time;
    }
    printf("***cublas Gemm Testing End***\n\n");
    fclose(fd);
    printf("***Encoder Gemm Testing End***\n");

#ifdef SPARSITY_ENABLED
    bool do_sparse_test = false;
    if (prop.major == 8 && (prop.minor == 0 || prop.minor == 6)) {
        do_sparse_test = true;
    }
    if (do_sparse_test && sizeof(T) == sizeof(half)) {
        printf("***cusparseLt Gemm Testing Begin***\n");
        // only first 3 cases can be sparse
        const int spgemm_num = 3;
        if (!isAppend) {
            fd = fopen(SPGEMM_CONFIG, "w+");
        }
        else {
            fd = fopen(SPGEMM_CONFIG, "a+");
            std::vector<std::string> config;
            char line[1024];
            while (fgets(line, 1024, fd) != NULL) {
                config.push_back(std::string(line));
            }
            line_count = config.size();
            if (config.size() >= (MAX_CONFIG_NUM * spgemm_num + 1))  // 6 cublas/cublasLt, first row is not included
            {
                int startIdx = config.size() - ((MAX_CONFIG_NUM - 1) * spgemm_num);
                fclose(fd);
                fd = fopen(SPGEMM_CONFIG, "w+");
                fprintf(fd, "%s", config[0].c_str());
                for (uint i = startIdx; i < config.size(); i++) {
                    fprintf(fd, "%s", config[i].c_str());
                }
                line_count = config.size() - (spgemm_num + 3);
            }
        }
        if (line_count == 0) {
            fprintf(
                fd,
                "batch_size, seq_len, head_num, size_per_head dataType ### batchCount, m, n, k, algoId, exec_time\n");
        }
        cusparseLtHandle_t handle;
        CHECK_CUSPARSE(cusparseLtInit(&handle));
        cusparseOrder_t order = CUSPARSE_ORDER_COL;
        cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseComputeType compute_type = CUSPARSE_COMPUTE_16F;
        unsigned alignment = 16;
        cudaStream_t stream = 0;
        float alpha2 = 1.0f;
        float beta2 = 0.0f;
        for (int i = 0; i < spgemm_num; ++i) {
            // to be compatable with spgemm wrapper, we let A be the weight matrix
            // so m and n are swapped
            // A: mxk B: kxn C:mxn
            int m = N[i], n = M[i], k = K[i];
            printf("\n-----------------------------\n");
            printf("GEMM test %d: [M: %d, K: %d, N: %d]\n", i, m, k, n);
            T* d_A = (T*)buffer;
            T* d_B = d_A + m * k * batchCount[i];
            T* d_C = d_B + k * n * batchCount[i];
            T* dA_compressed;
            {
                cusparseLtMatDescriptor_t matA;
                CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
                    &handle, &matA, m, k, m, alignment, CUDA_R_16F, order, CUSPARSELT_SPARSITY_50_PERCENT))
                CHECK_CUSPARSE(
                    cusparseLtSpMMAPrune2(&handle, &matA, true, opA, d_A, d_A, CUSPARSELT_PRUNE_SPMMA_STRIP, stream))
                size_t compressed_size;
                CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&handle, &matA, &compressed_size))
                check_cuda_error(cudaMalloc((void**)&dA_compressed, compressed_size));
                CHECK_CUSPARSE(cusparseLtSpMMACompress2(&handle, &matA, true, opA, d_A, dA_compressed, stream))
            }

            float exec_time = 99999.0f;
            int fast_algo = 0;
            for (int alg = 0; alg < 4; ++alg) {
                cudaDeviceSynchronize();
                cusparseLtMatDescriptor_t matA, matB, matC;
                void* d_workspace = nullptr;
                int num_streams = 1;
                cudaStream_t streams[1] = {stream};
                CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
                    &handle, &matA, m, k, m, alignment, CUDA_R_16F, order, CUSPARSELT_SPARSITY_50_PERCENT))
                CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, k, alignment, CUDA_R_16F, order))
                CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, m, n, m, alignment, CUDA_R_16F, order))
                gettimeofday(&start, NULL);
                for (int ite = 0; ite < ites; ++ite) {
                    // initializing MatDesc takes a lot of time
                    // and these descs can be stored to other place
                    // whereas storing MatMulPlan to other place will cause errors
                    cusparseLtMatmulDescriptor_t matmul;
                    cusparseLtMatmulAlgSelection_t alg_sel;
                    cusparseLtMatmulPlan_t plan;
                    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
                        &handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type))
                    CHECK_CUSPARSE(
                        cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
                    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
                        &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
                    size_t workspace_size;
                    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))
                    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size))
                    CHECK_CUSPARSE(cusparseLtMatmul(&handle,
                                                    &plan,
                                                    &alpha2,
                                                    dA_compressed,
                                                    d_B,
                                                    &beta2,
                                                    d_C,
                                                    d_C,
                                                    d_workspace,
                                                    streams,
                                                    num_streams))
                    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
                }
                cudaDeviceSynchronize();
                gettimeofday(&end, NULL);
                printf("algo_%d costs %.3fms \n", alg, diffTime(start, end) / ites);
                if (diffTime(start, end) < exec_time) {
                    exec_time = diffTime(start, end);
                    fast_algo = alg;
                }
            }
            exec_time /= ites;
            if (exec_time >= exec_times[i]) {
                fast_algo = -1;
            }
            printf("fast_algo %d\n", fast_algo);
            fprintf(fd,
                    "%d %d %d %d %d ### %d %d %d %d %d %f\n",
                    batch_size,
                    seq_len,
                    head_num,
                    size_per_head,
                    HALF_DATATYPE,
                    batchCount[i],
                    m,
                    n,
                    k,
                    fast_algo,
                    exec_time);
            cudaFree(dA_compressed);
        }
        CHECK_CUSPARSE(cusparseLtDestroy(&handle))
        fclose(fd);
        printf("***cusparseLt Gemm Testing End***\n");
    }
#endif
    printf("time list:");
    for (int i = 0; i < gemm_num; ++i) {
        printf("%.6f ", exec_times[i]);
    }
    printf("\n");
}

template void generate_sv_gemm_config<float>(
    int batch_size, int seq_len, int head_num, int size_per_head, void* buffer, bool isAppend);
template void generate_sv_gemm_config<half>(
    int batch_size, int seq_len, int head_num, int size_per_head, void* buffer, bool isAppend);

#ifdef ENABLE_BF16
template void generate_sv_gemm_config<__nv_bfloat16>(
    int batch_size, int seq_len, int head_num, int size_per_head, void* buffer, bool isAppend);
#endif

}  // namespace fastertransformer