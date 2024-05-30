
#include "src/fastertransformer/models/sv/SV.h"
#include "src/fastertransformer/models/sv/SVLayerWeightDebug.h"
#include "src/fastertransformer/models/sv/helper_file.h"

#include "stdio.h"
#include "stdlib.h"
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>

using namespace fastertransformer;
using namespace std;

/// all feats are fp32, and format are kLINEAR

template<typename T>
void test(size_t batch_size, size_t seq_len, size_t embed_dim, size_t head_num, size_t layer_num)
{
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));

    //
    // [ERROR] vit_gemm batch_size seq_len embed_dim head_number with_cls_token data_type int8_mode
    // e.g. ./build/bin/sv_gemm 1 512 256 8 1 1 0
    //
    std::vector<std::string> choices{
#ifdef NV_ORIN
        // "1 512 8 32 1 ### 1 256 512 256 6 0 18 0 1 0 0 12 0.008948",
        // "1 512 8 32 1 ### 1 512 24 4 23 0 5 3 0 1 0 7 0.005013",
        // "1 512 8 32 1 ### 1 24 512 256 21 0 11 1 0 0 0 13 0.006933",
        // "1 512 8 32 1 ### 1 256 512 3 0 0 14 0 0 0 0 0 0.007407",
        // "1 512 8 32 1 ### 1 512 512 256 6 0 20 1 1 0 0 11 0.011434",
        // "1 512 8 32 1 ### 1 256 512 512 6 0 18 0 0 0 0 12 0.011358",
        // "1 512 8 32 1 ### 1 8 512 256 21 0 5 1 0 0 0 20 0.006328",
        // "1 512 8 32 1 ### 1 5 512 256 24 0 11 0 0 0 0 20 0.007329",
        // "1 512 8 32 1 ### 8 512 512 32 113 -1 -1 -1 -1 -1 -1 -1 0.030930",
        // "1 512 8 32 1 ### 8 32 512 512 104 -1 -1 -1 -1 -1 -1 -1 0.025400"

        "1 512 8 32 1 ### 1 256 512 256 6 0 18 0 1 0 0 12 0.008974",
        "1 512 8 32 1 ### 1 512 48 4 23 0 5 6 0 1 0 7 0.005198",
        "1 512 8 32 1 ### 1 48 512 256 21 0 11 1 0 0 0 13 0.007429",
        "1 512 8 32 1 ### 1 256 512 3 24 0 15 1 0 0 0 0 0.007418",
        "1 512 8 32 1 ### 1 512 512 256 6 0 20 0 1 0 0 11 0.011486",
        "1 512 8 32 1 ### 1 256 512 512 6 0 18 0 1 0 0 12 0.011330",
        "1 512 8 32 1 ### 1 8 512 256 21 0 11 1 0 0 0 20 0.006383",
        "1 512 8 32 1 ### 1 5 512 256 24 0 11 0 0 0 0 20 0.007382",
        "1 512 8 32 1 ### 8 512 512 32 104 -1 -1 -1 -1 -1 -1 -1 0.035050",
        "1 512 8 32 1 ### 8 32 512 512 100 -1 -1 -1 -1 -1 -1 -1 0.024230"
#else
        "1 512 8 32 1 ### 1 256 512 256 6 0 15 1 1 0 0 17 0.006062",
        "1 512 8 32 1 ### 1 512 24 4 23 0 11 1 0 0 0 7 0.002662",
        "1 512 8 32 1 ### 1 24 512 256 21 0 15 3 0 4 73728 12 0.006707",
        "1 512 8 32 1 ### 1 256 512 3 0 0 14 0 0 0 0 0 0.003498",
        "1 512 8 32 1 ### 1 512 512 256 21 0 18 1 0 0 0 12 0.007485",
        "1 512 8 32 1 ### 1 256 512 512 6 0 15 1 1 0 0 17 0.007547",
        "1 512 8 32 1 ### 1 8 512 256 21 0 5 3 0 4 24576 20 0.005222",
        "1 512 8 32 1 ### 1 5 512 256 24 0 11 1 0 0 0 20 0.004239",
        "1 512 8 32 1 ### 8 512 512 32 105 -1 -1 -1 -1 -1 -1 -1 0.016810",
        "1 512 8 32 1 ### 8 32 512 512 109 -1 -1 -1 -1 -1 -1 -1 0.016800"
#endif
    };
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap(choices);

    std::mutex* cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper* cublas_wrapper =
        new cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, nullptr);

    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }
    const bool with_cls_token = true;
    const size_t inter_size = 512;  // ffn inner
    const size_t head_dim = embed_dim / head_num;

    size_t l2i_matr_h = 4;
    size_t l2i_matr_w = 4;
    int img_shape[2]{288, 736};  // !!!

    float pc_range[6]{-51.2, -51.2, -5.0, 51.2, 51.2, 3.0};
    size_t max_batch = batch_size;
    size_t num_cams = 6;  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    size_t ch = 256;
    size_t num_points = 1;
    size_t num_levels = 4;

    size_t num_classes = 8;  // !!!
    size_t num_reg_points = 8;
    int wp_ite = 1, ite = 0;  // 1000;

    cudaEvent_t inner_event_;
    cudaEvent_t event_base_;
    check_cuda_error(cudaEventCreate(&inner_event_));
    check_cuda_error(cudaEventCreate(&event_base_));

    SVWeight<T> params = SVWeight<T>(
        embed_dim, inter_size, layer_num, seq_len, num_classes, num_cams, num_points, num_levels, with_cls_token);

    const std::vector<const char*> cnn_feats_names{"cnn.out.feat1.%d-256-72-184",
                                                   "cnn.out.feat2.%d-256-36-92",
                                                   "cnn.out.feat3.%d-256-18-46",
                                                   "In.lidar2img.%d-4-4"};
    std::vector<std::string> cnn_feats_names_str;
    for (const auto& it : cnn_feats_names) {
        char str_buf[1024]{0};
        sprintf(str_buf, it, num_cams);
        cnn_feats_names_str.emplace_back(str_buf);
    }

    std::string data_in_path, data_out_path;
    if (num_cams == 12) {
        Load<T>(params, "1117_736_288_data/weights_1219/");
        data_in_path = "1117_736_288_data/tf_decoder_node_data_1219/";
    }
    else if (num_cams == 6) {
        Load<T>(params, "0822_736_288_data/weights_0915/");
        data_in_path = "0822_736_288_data/tf_decoder_node_data_0915/";
    }
    else if (num_cams == 4) {
        Load<T>(params, "0413_736_288_avod/weights/");
        data_in_path = "0413_736_288_avod/tf_decoder_node_data/";

        float __pc_range[6]{-20.200000, -20.200000, -5.000000, 20.200000, 20.200000, 3.000000};
        memcpy(pc_range, __pc_range, sizeof(pc_range));
    }

    data_out_path = "sv_example_fp32_linear_nc" + std::to_string(num_cams) + "/";

    fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
    cudaStream_t rt_stream;
    check_cuda_error(cudaStreamCreate(&rt_stream));

    SVTransformer<T>* sv = new SVTransformer<T>(max_batch,
                                                seq_len,
                                                embed_dim,
                                                head_num,
                                                inter_size,
                                                layer_num,
                                                num_cams,
                                                l2i_matr_h,
                                                l2i_matr_w,
                                                img_shape,
                                                pc_range,
                                                with_cls_token,
                                                getSMVersion(),
                                                1.0f,
                                                &params,
                                                stream,
                                                cublas_wrapper,
                                                &allocator,
                                                false);

    T *value1, *value2, *value3;
    deviceMalloc(&value1, batch_size * num_cams * ch * 72 * 184, false);
    deviceMalloc(&value2, batch_size * num_cams * ch * 36 * 92, false);
    deviceMalloc(&value3, batch_size * num_cams * ch * 18 * 46, false);
    /////////////////////
    float* lidar2img;
    deviceMalloc(&lidar2img, batch_size * num_cams * 4 * 4, false);

    float* pol_datas;
    deviceMalloc(&pol_datas, batch_size * num_cams * 5, false);

    float* cxy_cropxseyse_oxy;
    deviceMalloc(&cxy_cropxseyse_oxy, batch_size * num_cams * 8, false);

    //////////////////////
    float* last_reg_out;
    deviceMalloc(&last_reg_out, batch_size * seq_len * num_reg_points, false);
    float* last_cls_out;
    deviceMalloc(&last_cls_out, batch_size * seq_len * num_classes, false);
    ////// IN
    // value[4]   float32     [num_cams, ch, h, w]                         here is 4
    // lidar2img  float32     [num_cams, 4, 4]                                     5

    ////// OUT
    //
    //  last_reg_out   float32    [1, seq_len_, 8]
    //  last_cls_out   float32    [1, seq_len_, 5]
    for (int idx = 1; idx < 2; ++idx) {
        std::string inpp = data_in_path + std::to_string(idx) + "/";
        std::string outpp = data_out_path + std::to_string(idx) + "/";
        cudaacc::Read2Dptr(inpp + cnn_feats_names_str[0], {int(batch_size), int(num_cams), int(ch), 72, 184}, value1);
        cudaacc::Read2Dptr(inpp + cnn_feats_names_str[1], {int(batch_size), int(num_cams), int(ch), 36, 92}, value2);
        cudaacc::Read2Dptr(inpp + cnn_feats_names_str[2], {int(batch_size), int(num_cams), int(ch), 18, 46}, value3);
        cudaacc::Read2Dptr(inpp + cnn_feats_names_str[3], {int(batch_size), int(num_cams), 4, 4}, lidar2img);

        std::vector<Tensor> input_tensors{
            Tensor{MEMORY_GPU,
                   getTensorType<T>(),
                   std::vector<size_t>{num_cams, ch, 72, 184},
                   value1,
                   1.f,
                   TensorFormat::kLINEAR},
            Tensor{MEMORY_GPU,
                   getTensorType<T>(),
                   std::vector<size_t>{num_cams, ch, 36, 92},
                   value2,
                   1.f,
                   TensorFormat::kLINEAR},
            Tensor{MEMORY_GPU,
                   getTensorType<T>(),
                   std::vector<size_t>{num_cams, ch, 18, 46},
                   value3,
                   1.f,
                   TensorFormat::kLINEAR},
            Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{num_cams, 4, 4}, lidar2img}};

        /////////////////////////////////////////////////////////////////////////////////////
        if (num_cams == 4) {  /// avod
            cudaacc::Read2Dptr(inpp + "In.pol_datas.4-5", {int(num_cams), 5}, pol_datas);
            cudaacc::Read2Dptr(inpp + "In.cxy_cropxseyse_oxy.4-8", {int(num_cams), 8}, cxy_cropxseyse_oxy);

            input_tensors.emplace_back(
                Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{num_cams, 5ul}, pol_datas});
            input_tensors.emplace_back(
                Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{num_cams, 8ul}, cxy_cropxseyse_oxy});
        }

        std::vector<Tensor> output_tensors{Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  std::vector<size_t>{batch_size, seq_len, num_reg_points},
                                                  last_reg_out},
                                           Tensor{MEMORY_GPU,
                                                  getTensorType<float>(),
                                                  std::vector<size_t>{batch_size, seq_len, num_classes},
                                                  last_cls_out}};

        APP_PRINTF("warm up %d times, forward %d times\n", wp_ite, ite);
        for (int i = 0; i < wp_ite; ++i) {
            sv->forward(&output_tensors, &input_tensors, rt_stream);
        }
        cudaStreamSynchronize(rt_stream);
        APP_PRINTF("begin to save...\n");

        cudaacc::mkdir_p(outpp.c_str());
        cudaacc::WriteFromDptr(outpp + "Out.outputs_bbox.1-512-8", {int(batch_size), int(seq_len), int(num_reg_points)}, last_reg_out);
        cudaacc::WriteFromDptr(outpp + "Out.outputs_scores.1-512-8", {int(batch_size), int(seq_len), int(num_classes)}, last_cls_out);
        APP_PRINTF("warm up done, idx %d\n", idx);

        CudaTimer cuda_timer(rt_stream);
        cuda_timer.start();
        for (int i = 0; i < ite; i++) {
            sv->forward(&output_tensors, &input_tensors, rt_stream);
        }
        float total_time = cuda_timer.stop();

        APP_PRINTF("forward done\n");
        APP_PRINTF("SVT-CPP-time %.3f ms (%d iterations)\n", ite > 0 ? total_time / ite : 0, ite);
    }
    delete sv;
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    check_cuda_error(cudaFree(value1));
    check_cuda_error(cudaFree(value2));
    check_cuda_error(cudaFree(value3));
    check_cuda_error(cudaFree(lidar2img));
    check_cuda_error(cudaFree(pol_datas));
    check_cuda_error(cudaFree(cxy_cropxseyse_oxy));

    check_cuda_error(cudaFree(last_cls_out));
    check_cuda_error(cudaFree(last_reg_out));

    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cublasLtDestroy(cublaslt_handle));

    check_cuda_error(cudaDeviceSynchronize());
    check_cuda_error(cudaGetLastError());

    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaStreamDestroy(rt_stream));

    check_cuda_error(cudaEventDestroy(inner_event_));
    check_cuda_error(cudaEventDestroy(event_base_));
}

int main(int argc, char* argv[])
{
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    APP_PRINTF("Device %s\n", prop.name);

    const size_t batch_size = 1;
    const size_t seq_len = 512;
    const size_t embed_dim = 256;
    const size_t head_num = 8;
    const size_t layer_num = 4;

    test<float>(batch_size, seq_len, embed_dim, head_num, layer_num);

    APP_PRINTF("Main Done\n");
#ifdef NV_ORIN
    APP_PRINTF("---NV_ORIN--- \n");
#endif
    return 0;
}
