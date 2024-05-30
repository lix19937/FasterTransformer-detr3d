/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "helper_string.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasAlgoMap.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include <cuda_profiler_api.h>

using namespace fastertransformer;

//  cmake -DSM=86 -DCMAKE_BUILD_TYPE=Release ..  && make 

// /home/igs/transformer/FasterTransformer-main/src/fastertransformer/kernels/decoder_masked_multihead_attention.h
// batch_size        beam_width head_num size_per_head inter_size vocab_size num_layers max_seq_len memory_max_seq_len
// memory_hidden_units top_k top_p is_fp16
/*
 ./bin/decode_ma 1   1 8        32            2048        30000     1          900          900               256 0     0.6   1
*/
template<typename T>
int test()
{
    int head_num = 8;
    int local_head_num = head_num;
    int size_per_head = 32;
    int batch_size = 1;
    bool is_free_buffer_after_forward = false;
    int hidden_units = head_num * size_per_head;  // 256
    int seq_len = 900;
    int max_batch_size = batch_size;
    int max_seq_len = seq_len;
    int d_model = hidden_units;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    check_cuda_error(cudaStreamCreate(&stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in");
    Allocator<AllocatorType::CUDA> allocator(getDevice());
    std::mutex* cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
    if (std::is_same<T, half>::value) {
        cublas_wrapper.setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    auto self_attention_layer = new DecoderSelfAttentionLayer<T>(
        max_batch_size, head_num, size_per_head, stream, &cublas_wrapper, &allocator, is_free_buffer_after_forward);

    AttentionWeight<T> attention_weights;

    /// read weight(bias)
    const std::string bp = "/home/igs/transformer/FasterTransformer-main/debug_ma_data/";
    std::vector<std::string> ma_input_file{
        bp + "input_key_900x256.txt", bp + "input_query_900x256.txt", bp + "input_value_900x256.txt"};

    // std::vector<std::string> ma_output_file{bp+"attn_output_900x256.txt"};

    std::vector<std::string> qkv_file{bp + "query_bias_1x256.txt",
                                      bp + "query_weight_256x256.txt",
                                      bp + "key_bias_1x256.txt",
                                      bp + "key_weight_256x256.txt",
                                      bp + "value_bias_1x256.txt",
                                      bp + "value_weight_256x256.txt",
                                      bp + "outproj_bias_1x256.txt",
                                      bp + "outproj_weight_256x256.txt"};

    print_mem_usage();

    LOG_INFO("file_name1");
    T *q_kernel = NULL, *q_bias = NULL, *k_kernel = NULL, *k_bias = NULL, *v_kernel = NULL, *v_bias = NULL,
      *linear_kernel = NULL, *linear_bias = NULL;
    check_cuda_error(cudaMalloc(&q_kernel, hidden_units * hidden_units * sizeof(T)));
    check_cuda_error(cudaMalloc(&q_bias, hidden_units * batch_size * sizeof(T)));
    check_cuda_error(cudaMalloc(&k_kernel, hidden_units * hidden_units * sizeof(T)));
    check_cuda_error(cudaMalloc(&k_bias, hidden_units * batch_size * sizeof(T)));
    check_cuda_error(cudaMalloc(&v_kernel, hidden_units * hidden_units * sizeof(T)));
    check_cuda_error(cudaMalloc(&v_bias, hidden_units * batch_size * sizeof(T)));
    check_cuda_error(cudaMalloc(&linear_kernel, hidden_units * hidden_units * sizeof(T)));
    check_cuda_error(cudaMalloc(&linear_bias, hidden_units * batch_size * sizeof(T)));

    LOG_INFO("file_name1");
    cuda_ops::Read2Buff_GPU(qkv_file[0], {batch_size, hidden_units}, q_bias);
    cuda_ops::Read2Buff_GPU(qkv_file[1], {hidden_units, hidden_units}, q_kernel);

    LOG_INFO("file_name2");
    cuda_ops::Read2Buff_GPU(qkv_file[2], {batch_size, hidden_units}, k_bias);
    cuda_ops::Read2Buff_GPU(qkv_file[3], {hidden_units, hidden_units}, k_kernel);

    LOG_INFO("file_name3");
    cuda_ops::Read2Buff_GPU(qkv_file[4], {batch_size, hidden_units}, v_bias);
    cuda_ops::Read2Buff_GPU(qkv_file[5], {hidden_units, hidden_units}, v_kernel);

    LOG_INFO("file_name4");
    cuda_ops::Read2Buff_GPU(qkv_file[6], {batch_size, hidden_units}, linear_bias);
    cuda_ops::Read2Buff_GPU(qkv_file[7], {hidden_units, hidden_units}, linear_kernel);
    LOG_INFO("load txt done");

    attention_weights.query_weight.bias = q_bias;
    attention_weights.query_weight.kernel = q_kernel;

    attention_weights.key_weight.bias = k_bias;
    attention_weights.key_weight.kernel = k_kernel;

    attention_weights.value_weight.bias = v_bias;
    attention_weights.value_weight.kernel = v_kernel;

    attention_weights.attention_output_weight.bias = linear_bias;
    attention_weights.attention_output_weight.kernel = linear_kernel;

    /// assign input tensor
    T* d_attention_input;
    int d_model_len = seq_len * batch_size * hidden_units * 3;
    check_cuda_error(cudaMalloc(&d_attention_input, d_model_len * sizeof(T)));
    cuda_ops::Read2Buff_GPU(ma_input_file[0], {seq_len, hidden_units}, d_attention_input);
    cuda_ops::Read2Buff_GPU(ma_input_file[1], {seq_len, hidden_units}, d_attention_input + hidden_units * seq_len);
    cuda_ops::Read2Buff_GPU(ma_input_file[2], {seq_len, hidden_units}, d_attention_input + hidden_units * seq_len * 2);

    int* d_sequence_lengths;
    check_cuda_error(cudaMalloc(&d_sequence_lengths, batch_size * sizeof(int)));
    std::vector<int> h_sequence_lengths(batch_size, seq_len);
    cudaH2Dcpy(d_sequence_lengths, h_sequence_lengths.data(), batch_size);

    /// assign output tensor
    T* d_attention_output;
    T* d_key_cache;
    T* d_value_cache;
    check_cuda_error(cudaMalloc(&d_attention_output, hidden_units * seq_len * batch_size * sizeof(T)));
    check_cuda_error(cudaMalloc(&d_key_cache, batch_size * hidden_units* sizeof(T)));
    check_cuda_error(cudaMalloc(&d_value_cache, batch_size * hidden_units* sizeof(T)));

    print_mem_usage();

    ////////////////////////////////////////////////////
    // input tensors:
    //      attention_input [batch_size, d_model_],
    //      finished [batch_size],
    //      sequence_lengths [batch_size]
    //      input_lengths [batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu
    //      cache_indirection [batch_size / beam_width, beam_width, max_seq_len]
    //      relative_attention_bias [1, head_num, step, step] or [1, head_num, max_seq_len, max_seq_len] (option)
    // output tensors:
    //      attention_output [batch_size, d_model_],
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]

    int __tmp = 0, __step = 1;
    auto data_type = getTensorType<T>();
    std::vector<Tensor> self_attention_input_tensors{
        Tensor{MEMORY_GPU, data_type, {size_t(batch_size), size_t(d_model)}, d_attention_input},  // attention_input
        Tensor{MEMORY_GPU, TYPE_BOOL, {size_t(batch_size)}, nullptr},                                 // finished
        Tensor{MEMORY_GPU, TYPE_INT32, {size_t(batch_size)}, d_sequence_lengths},            // sequence_lengths
        Tensor{MEMORY_GPU, TYPE_INT32, {size_t(batch_size)}, nullptr},                       // input_lengths
        Tensor{MEMORY_CPU, TYPE_INT32, {1ul}, &__tmp},                                       // max_input_length
        Tensor{MEMORY_CPU, TYPE_INT32, {1ul}, &__step},                                      // step
        Tensor{MEMORY_GPU, TYPE_INT32, {size_t(batch_size), 1ul, size_t(seq_len)}, nullptr}  // cache_indirection
    };

    std::vector<Tensor> self_attention_output_tensors{
        Tensor{MEMORY_GPU, data_type, {size_t(batch_size), size_t(d_model)}, d_attention_output},
        Tensor{MEMORY_GPU,
               data_type,
               {size_t(batch_size),
                size_t(local_head_num),
                size_t(size_per_head / head_num),
                size_t(max_seq_len),
                size_t(head_num)},
               d_key_cache},
        Tensor{MEMORY_GPU,
               data_type,
               {size_t(batch_size), size_t(local_head_num), size_t(max_seq_len), size_t(size_per_head)},
               d_value_cache}};

    LOG_INFO("forward ...");

    self_attention_layer->forward(&self_attention_output_tensors, &self_attention_input_tensors, &attention_weights);

    LOG_INFO("forward done\n");

    std::vector<T> h_atten_out(900 * 256);
    // cudaD2Hcpy(h_atten_out.data(), d_attention_output, h_atten_out.size());

    cuda_ops::writerp("./atten_out.txt", h_atten_out.data(), 900, 1, 256, 2);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    delete self_attention_layer;
    
    print_mem_usage();

    // if (q_kernel != nullptr){
    //   LOG_INFO("xxxxxxxxxxxxxx cudaFree ... ");
    //   check_cuda_error(cudaFree(q_kernel));
    //   q_kernel = nullptr;
    //   LOG_INFO("xxxxxxxxxxxxxx cudaFree done");
    // }else{
    //   LOG_INFO("yyyyyyyyyyyyy cudaFree done");
    // }

    // LOG_INFO("q_kernel ...");
    // deviceFree(q_kernel);
    // LOG_INFO("q_bias ...");
    // deviceFree(q_bias);
    // LOG_INFO("k_kernel ...");
    // deviceFree(k_kernel);
    // LOG_INFO("k_bias ...");
    // deviceFree(k_bias);
    // LOG_INFO("v_kernel ...");
    // deviceFree(v_kernel);
    // LOG_INFO("v_bias ...");
    // deviceFree(v_bias);
    // LOG_INFO("linear_kernel ...");
    // deviceFree(linear_kernel);
    // LOG_INFO("linear_bias ...");
    // deviceFree(linear_bias);

    // LOG_INFO("d_attention_input ...");
    // deviceFree(d_attention_input);
    // LOG_INFO("d_sequence_lengths ...");
    // deviceFree(d_sequence_lengths);
    // LOG_INFO("d_attention_output ...");
    // deviceFree(d_attention_output);
    // LOG_INFO("d_key_cache ...");
    // deviceFree(d_key_cache);
    // LOG_INFO("d_value_cache ...");
    // deviceFree(d_value_cache);

    cudaStreamDestroy(stream);
    return 0;
}

int main(int argc, char** argv)
{
    for (int i = 0; i < 1; ++i) {
        LOG_INFO(" ===>idx:%d", i);
        test<float>();
    }

    return 0;
}
