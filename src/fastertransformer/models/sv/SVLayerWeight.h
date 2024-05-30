/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#pragma once

#include "helper_macros.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

#define WEIGHT_N (36)

template<typename T, typename SRC_T = float>
struct SVLayerWeight {

    SVLayerWeight() = delete;
    SVLayerWeight(const int embed_dim,
                  const int ffn_inter_size,
                  int layer_idx,
                  const int num_cams,
                  const int num_points,
                  const int num_levels,
                  const bool hold_buffer):
        embed_dim_(embed_dim),
        ffn_inter_size_(ffn_inter_size),
        layer_idx_(layer_idx),
        num_cams_(num_cams),
        num_points_(num_points),
        num_levels_(num_levels)
    {
        // mha
        weights_size_[0] = embed_dim_ * embed_dim_;
        weights_size_[1] = 1 * embed_dim_;
        weights_size_[2] = embed_dim_ * embed_dim_;
        weights_size_[3] = 1 * embed_dim_;
        weights_size_[4] = embed_dim_ * embed_dim_;
        weights_size_[5] = 1 * embed_dim_;
        weights_size_[6] = embed_dim_ * embed_dim_;
        weights_size_[7] = 1 * embed_dim_;

        // norm
        weights_size_[8] = 1 * embed_dim_;
        weights_size_[9] = 1 * embed_dim_;

        // ca    self.num_cams  =12, self.num_points =1, self.num_levels  = 4
        weights_size_[10] = embed_dim_ * num_cams_ * num_points_ * num_levels_;  // 12,1,4
        weights_size_[11] = 1 * num_cams_ * num_points_ * num_levels_;

        weights_size_[12] = embed_dim_ * embed_dim_;
        weights_size_[13] = 1 * embed_dim_;

        weights_size_[14] = 3 * embed_dim_;
        weights_size_[15] = embed_dim_;

        weights_size_[16] = 1 * embed_dim_;
        weights_size_[17] = 1 * embed_dim_;

        weights_size_[18] = embed_dim_ * embed_dim_;
        weights_size_[19] = 1 * embed_dim_;

        weights_size_[20] = 1 * embed_dim_;
        weights_size_[21] = 1 * embed_dim_;

        // norm
        weights_size_[22] = 1 * embed_dim_;
        weights_size_[23] = 1 * embed_dim_;

        // ffn
        weights_size_[24] = embed_dim_ * ffn_inter_size_;
        weights_size_[25] = ffn_inter_size_;

        weights_size_[26] = ffn_inter_size_ * embed_dim_;
        weights_size_[27] = embed_dim_;

        // norm
        weights_size_[28] = embed_dim_;
        weights_size_[29] = embed_dim_;

        // reg_branches
        weights_size_[30] = embed_dim_ * embed_dim_;
        weights_size_[31] = embed_dim_;
        weights_size_[32] = embed_dim_ * embed_dim_;
        weights_size_[33] = embed_dim_;
        weights_size_[34] = embed_dim_ * 8;
        weights_size_[35] = 8;

        if (hold_buffer) {
            for (int i = 0; i < WEIGHT_N; i++) {
                deviceMalloc(&weights_ptr_[i], weights_size_[i]);
            }

            setWeightPtr();
        }
    }

    ~SVLayerWeight()
    {
        if (is_maintain_buffer_) {
            for (int i = 0; i < WEIGHT_N; i++) {
                deviceFree(weights_ptr_[i]);
            }

            mha_weights.query_weight.kernel = nullptr;
            mha_weights.query_weight.bias = nullptr;
            mha_weights.key_weight.kernel = nullptr;
            mha_weights.key_weight.bias = nullptr;
            mha_weights.value_weight.kernel = nullptr;
            mha_weights.value_weight.bias = nullptr;
            mha_weights.attention_output_weight.kernel = nullptr;
            mha_weights.attention_output_weight.bias = nullptr;

            mha_norm_weights.gamma = nullptr;
            mha_norm_weights.beta = nullptr;

            ca_weights.attention_weights.kernel = nullptr;
            ca_weights.attention_weights.bias = nullptr;
            ca_weights.output_proj.kernel = nullptr;
            ca_weights.output_proj.bias = nullptr;
            ca_weights.position_encoder_fc1.kernel = nullptr;
            ca_weights.position_encoder_fc1.bias = nullptr;
            ca_weights.position_encoder_ln1.kernel = nullptr;
            ca_weights.position_encoder_ln1.bias = nullptr;
            ca_weights.position_encoder_fc2.kernel = nullptr;
            ca_weights.position_encoder_fc2.bias = nullptr;
            ca_weights.position_encoder_ln2.kernel = nullptr;
            ca_weights.position_encoder_ln2.bias = nullptr;

            ca_norm_weights.gamma = nullptr;
            ca_norm_weights.beta = nullptr;

            ffn_weights.intermediate_weight.kernel = nullptr;
            ffn_weights.intermediate_weight.bias = nullptr;
            ffn_weights.output_weight.kernel = nullptr;
            ffn_weights.output_weight.bias = nullptr;

            ffn_norm_weights.gamma = nullptr;
            ffn_norm_weights.beta = nullptr;

            reg_weights.fc1.kernel = nullptr;
            reg_weights.fc1.bias = nullptr;
            reg_weights.fc2.kernel = nullptr;
            reg_weights.fc2.bias = nullptr;
            reg_weights.fc3.kernel = nullptr;
            reg_weights.fc3.bias = nullptr;

            is_maintain_buffer_ = false;
        }
    }

    SVLayerWeight(const SVLayerWeight& other): embed_dim_(other.embed_dim_), ffn_inter_size_(other.ffn_inter_size_)
    {
        memcpy(weights_size_, other.weights_size_, sizeof(size_t) * WEIGHT_N);
        layer_idx_ = other.layer_idx_;
        if (other.is_maintain_buffer_) {
            for (int i = 0; i < WEIGHT_N; i++) {
                if (!is_maintain_buffer_) {
                    deviceMalloc(&weights_ptr_[i], weights_size_[i]);
                }
                cudaD2Dcpy(weights_ptr_[i], other.weights_ptr_[i], weights_size_[i]);
            }

            setWeightPtr();
        }
    }

    SVLayerWeight& operator=(const SVLayerWeight& other)
    {
        embed_dim_ = other.embed_dim_;
        ffn_inter_size_ = other.ffn_inter_size_;
        layer_idx_ = other.layer_idx_;
        memcpy(weights_size_, other.weights_size_, sizeof(size_t) * WEIGHT_N);
        if (other.is_maintain_buffer_) {
            for (int i = 0; i < WEIGHT_N; ++i) {
                if (!is_maintain_buffer_) {
                    deviceMalloc(&weights_ptr_[i], weights_size_[i]);
                }
                cudaD2Dcpy(weights_ptr_[i], other.weights_ptr_[i], weights_size_[i]);
            }

            setWeightPtr();
        }

        return *this;
    }

    inline size_t GetWeightCount()
    {
        return WEIGHT_N;
    }

    size_t GetSerializeSize()
    {
        size_t count{0};
        for (int i = 0; i < WEIGHT_N; ++i) {
            count += weights_size_[i];
        }
        if (std::is_same<SRC_T, float>::value) {
            return sizeof(SRC_T) * count;
        }

        return sizeof(T) * count;
    }

    void Hfp32ToDhalf(const float* w, half* h_ptr, const void* d_ptr, const int idx)
    {
        for (size_t i = 0; i < weights_size_[idx]; ++i) {
            h_ptr[i] = w[i];
        }

        cudaMemcpy((half*)d_ptr, h_ptr, sizeof(half) * weights_size_[idx], cudaMemcpyHostToDevice);
    }

    void CopyWeightsFromHostBuffersFp32ToDeviceHalf(const float* const*& w)
    {
        half* h_ptr = (half*)malloc(getworkspace() * sizeof(half));

        int i = 0;

        // mha
        Hfp32ToDhalf(*w++, h_ptr, mha_weights.query_weight.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, mha_weights.query_weight.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, mha_weights.key_weight.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, mha_weights.key_weight.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, mha_weights.value_weight.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, mha_weights.value_weight.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, mha_weights.attention_output_weight.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, mha_weights.attention_output_weight.bias, i++);

        // mha_norm
        Hfp32ToDhalf(*w++, h_ptr, mha_norm_weights.gamma, i++);
        Hfp32ToDhalf(*w++, h_ptr, mha_norm_weights.beta, i++);
        
        // ca
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.attention_weights.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.attention_weights.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.output_proj.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.output_proj.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.position_encoder_fc1.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.position_encoder_fc1.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.position_encoder_ln1.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.position_encoder_ln1.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.position_encoder_fc2.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.position_encoder_fc2.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.position_encoder_ln2.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_weights.position_encoder_ln2.bias, i++);

        // ca_norm
        Hfp32ToDhalf(*w++, h_ptr, ca_norm_weights.gamma, i++);
        Hfp32ToDhalf(*w++, h_ptr, ca_norm_weights.beta, i++);

        // ffn
        Hfp32ToDhalf(*w++, h_ptr, ffn_weights.intermediate_weight.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, ffn_weights.intermediate_weight.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, ffn_weights.output_weight.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, ffn_weights.output_weight.bias, i++);

        // ffn_norm
        Hfp32ToDhalf(*w++, h_ptr, ffn_norm_weights.gamma, i++);
        Hfp32ToDhalf(*w++, h_ptr, ffn_norm_weights.beta, i++);

        // reg_branch
        Hfp32ToDhalf(*w++, h_ptr, reg_weights.fc1.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, reg_weights.fc1.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, reg_weights.fc2.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, reg_weights.fc2.bias, i++);
        Hfp32ToDhalf(*w++, h_ptr, reg_weights.fc3.kernel, i++);
        Hfp32ToDhalf(*w++, h_ptr, reg_weights.fc3.bias, i);

        free(h_ptr);
    }

    void CopyWeightsFromHostBuffers(const T* const*& w)
    {
        // clang-format off
        // mha
        cudaMemcpy(const_cast<T*>(mha_weights.query_weight.kernel),*w++,sizeof(T) * weights_size_[0], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(mha_weights.query_weight.bias), *w++, sizeof(T) * weights_size_[1], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(mha_weights.key_weight.kernel), *w++, sizeof(T) * weights_size_[2], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(mha_weights.key_weight.bias), *w++, sizeof(T) * weights_size_[3], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(mha_weights.value_weight.kernel), *w++, sizeof(T) * weights_size_[4], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(mha_weights.value_weight.bias), *w++, sizeof(T) * weights_size_[5], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(mha_weights.attention_output_weight.kernel), *w++, sizeof(T) * weights_size_[6],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(mha_weights.attention_output_weight.bias),*w++,sizeof(T) * weights_size_[7],cudaMemcpyHostToDevice);

        // mha_norm
        cudaMemcpy(const_cast<T*>(mha_norm_weights.gamma), *w++, sizeof(T) * weights_size_[8], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(mha_norm_weights.beta), *w++, sizeof(T) * weights_size_[9], cudaMemcpyHostToDevice);

        // ca
        cudaMemcpy(const_cast<T*>(ca_weights.attention_weights.kernel), *w++,sizeof(T) * weights_size_[10],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.attention_weights.bias), *w++, sizeof(T) * weights_size_[11], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.output_proj.kernel), *w++, sizeof(T) * weights_size_[12], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.output_proj.bias), *w++, sizeof(T) * weights_size_[13], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.position_encoder_fc1.kernel), *w++,sizeof(T) * weights_size_[14],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.position_encoder_fc1.bias), *w++,sizeof(T) * weights_size_[15],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.position_encoder_ln1.kernel), *w++,sizeof(T) * weights_size_[16],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.position_encoder_ln1.bias), *w++,sizeof(T) * weights_size_[17],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.position_encoder_fc2.kernel), *w++,sizeof(T) * weights_size_[18], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.position_encoder_fc2.bias), *w++,sizeof(T) * weights_size_[19],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.position_encoder_ln2.kernel), *w++, sizeof(T) * weights_size_[20],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_weights.position_encoder_ln2.bias),*w++,sizeof(T) * weights_size_[21],cudaMemcpyHostToDevice);

        // ca_norm
        cudaMemcpy(const_cast<T*>(ca_norm_weights.gamma), *w++, sizeof(T) * weights_size_[22], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ca_norm_weights.beta), *w++, sizeof(T) * weights_size_[23], cudaMemcpyHostToDevice);

        // ffn
        cudaMemcpy(const_cast<T*>(ffn_weights.intermediate_weight.kernel), *w++,sizeof(T) * weights_size_[24],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ffn_weights.intermediate_weight.bias), *w++,sizeof(T) * weights_size_[25],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ffn_weights.output_weight.kernel), *w++, sizeof(T) * weights_size_[26],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ffn_weights.output_weight.bias), *w++, sizeof(T) * weights_size_[27], cudaMemcpyHostToDevice);

        // ffn_norm
        cudaMemcpy(const_cast<T*>(ffn_norm_weights.gamma), *w++, sizeof(T) * weights_size_[28], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ffn_norm_weights.beta), *w++, sizeof(T) * weights_size_[29], cudaMemcpyHostToDevice);

        // reg_branch
        cudaMemcpy(const_cast<T*>(reg_weights.fc1.kernel), *w++, sizeof(T) * weights_size_[30], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(reg_weights.fc1.bias), *w++, sizeof(T) * weights_size_[31], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(reg_weights.fc2.kernel), *w++, sizeof(T) * weights_size_[32], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(reg_weights.fc2.bias), *w++, sizeof(T) * weights_size_[33], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(reg_weights.fc3.kernel), *w++, sizeof(T) * weights_size_[34], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(reg_weights.fc3.bias), *w++, sizeof(T) * weights_size_[35], cudaMemcpyHostToDevice);
        // clang-format on
    }

    size_t getworkspace() const
    {
        size_t max_sz = 0;
        for (int i = 0; i < WEIGHT_N; ++i) {
            max_sz = std::max(max_sz, weights_size_[i]);
        }
        return max_sz;
    }

    void DhalfToHfp32(half* h_ptr, float* ptr, const int idx)
    {
        cudaMemcpy(h_ptr, weights_ptr_[idx], sizeof(half) * weights_size_[idx], cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < weights_size_[idx]; ++i) {
            ptr[i] = h_ptr[i];
        }
    }

    void serialize(void* buffer)
    {
        char* tmp_buf = (char*)buffer;

        if (std::is_same<T, half>::value && std::is_same<SRC_T, float>::value) {
            half* buf = (half*)malloc(getworkspace() * sizeof(half));
            for (int i = 0; i < WEIGHT_N; ++i) {
                DhalfToHfp32(buf, (float*)tmp_buf, i);
                tmp_buf += sizeof(SRC_T) * weights_size_[i];
            }
            free(buf);
            return;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////
        for (int i = 0; i < WEIGHT_N; ++i) {
            cudaMemcpy(tmp_buf, weights_ptr_[i], sizeof(T) * weights_size_[i], cudaMemcpyDeviceToHost);
            tmp_buf += sizeof(T) * weights_size_[i];
        }
    }

    void deserialize(const void* buffer)
    {
        if (!is_maintain_buffer_) {
            return;
        }

        char* tmp_buf = (char*)buffer;

        if (std::is_same<T, half>::value && std::is_same<SRC_T, float>::value) {
            T* buf = (T*)malloc(getworkspace() * sizeof(T));
            for (int i = 0; i < WEIGHT_N; ++i) {
                Hfp32ToDhalf((const float*)tmp_buf, (half*)buf, weights_ptr_[i], i);
                tmp_buf += sizeof(SRC_T) * weights_size_[i];
            }
            free(buf);
            return;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////
        for (int i = 0; i < WEIGHT_N; ++i) {
            cudaMemcpy(weights_ptr_[i], tmp_buf, sizeof(T) * weights_size_[i], cudaMemcpyHostToDevice);
            tmp_buf += sizeof(T) * weights_size_[i];
        }
    }

    void ExportWeights(int layer_idx) const
    {
        int i = 0;
        // clang-format off
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention.query.weight.256-256", {256,256},  weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention.query.bias.1-256", {1,256}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention.key.weight.256-256", {256,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention.key.bias.1-256", {1,256}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention.value.weight.256-256", {256,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention.value.bias.1-256", {1,256}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention.out.weight.256-256", {256,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention.out.bias.1-256", {1,256}, weights_ptr_[i++]);

      /// mha-ln
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention_norm.ln.weight.1-256", {1,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".mh_attention_norm.ln.bias.1-256", {1,256}, weights_ptr_[i++]);

      /// ca
      const int __dims3 = num_cams_ * num_points_ * num_levels_;
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.attention_weights.fc.weight.256-" + std::to_string(__dims3), {256,__dims3}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.attention_weights.fc.bias.1-" + std::to_string(__dims3), {1,__dims3}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.output_proj.fc.weight.256-256", {256,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.output_proj.fc.bias.1-256", {1,256}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.position_encoder.fc1.weight.3-256", {3,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.position_encoder.fc1.bias.1-256", {1,256}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.position_encoder.ln1.weight.1-256", {1,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.position_encoder.ln1.bias.1-256", {1,256}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.position_encoder.fc2.weight.256-256", {256,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.position_encoder.fc2.bias.1-256", {1,256}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.position_encoder.ln2.weight.1-256", {1,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention.position_encoder.ln2.bias.1-256", {1,256}, weights_ptr_[i++]);

      /// ca-ln
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention_norm.ln.weight.1-256", {1,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".cross_attention_norm.ln.bias.1-256", {1,256}, weights_ptr_[i++]);

      /// ffn
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".ffn.fc1.weight.256-512", {256,512}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".ffn.fc1.bias.1-512", {1,512}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".ffn.fc2.weight.512-256", {512,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".ffn.fc2.bias.1-256", {1,256}, weights_ptr_[i++]);

      /// ffn-ln
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".ffn_norm.ln.weight.1-256", {1,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".ffn_norm.ln.bias.1-256", {1,256}, weights_ptr_[i++]);

      /// reg_branches
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".reg_branches.fc1.weight.256-256", {256,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".reg_branches.fc1.bias.1-256", {1,256}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".reg_branches.fc2.weight.256-256", {256,256}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".reg_branches.fc2.bias.1-256", {1,256}, weights_ptr_[i++]);

      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".reg_branches.fc3.weight.256-8", {256,8}, weights_ptr_[i++]);
      cudaacc::WriteFromDptr("./from_plugin/block_"+ std::to_string(layer_idx+1) + ".reg_branches.fc3.bias.1-8", {1,8}, weights_ptr_[i]);
        // clang-format on
    }

    T*& GetPtr(const int i)
    {
        return weights_ptr_[i];
    }

    AttentionWeight<T> mha_weights;
    LayerNormWeight<T> mha_norm_weights;
    CAttentionWeight<T> ca_weights;
    LayerNormWeight<T> ca_norm_weights;
    FfnWeight<T> ffn_weights;
    LayerNormWeight<T> ffn_norm_weights;
    RegBranchWeight<T> reg_weights;

private:
    void setWeightPtr()
    {
        //
        mha_weights.query_weight.kernel = weights_ptr_[0];
        mha_weights.query_weight.bias = weights_ptr_[1];
        mha_weights.key_weight.kernel = weights_ptr_[2];
        mha_weights.key_weight.bias = weights_ptr_[3];
        mha_weights.value_weight.kernel = weights_ptr_[4];
        mha_weights.value_weight.bias = weights_ptr_[5];
        mha_weights.attention_output_weight.kernel = weights_ptr_[6];
        mha_weights.attention_output_weight.bias = weights_ptr_[7];

        //
        mha_norm_weights.gamma = weights_ptr_[8];
        mha_norm_weights.beta = weights_ptr_[9];

        //
        ca_weights.attention_weights.kernel = weights_ptr_[10];
        ca_weights.attention_weights.bias = weights_ptr_[11];
        ca_weights.output_proj.kernel = weights_ptr_[12];
        ca_weights.output_proj.bias = weights_ptr_[13];

        ca_weights.position_encoder_fc1.kernel = weights_ptr_[14];
        ca_weights.position_encoder_fc1.bias = weights_ptr_[15];
        ca_weights.position_encoder_ln1.kernel = weights_ptr_[16];
        ca_weights.position_encoder_ln1.bias = weights_ptr_[17];
        ca_weights.position_encoder_fc2.kernel = weights_ptr_[18];
        ca_weights.position_encoder_fc2.bias = weights_ptr_[19];
        ca_weights.position_encoder_ln2.kernel = weights_ptr_[20];
        ca_weights.position_encoder_ln2.bias = weights_ptr_[21];

        //
        ca_norm_weights.gamma = weights_ptr_[22];
        ca_norm_weights.beta = weights_ptr_[23];

        //
        ffn_weights.intermediate_weight.kernel = weights_ptr_[24];
        ffn_weights.intermediate_weight.bias = weights_ptr_[25];
        ffn_weights.output_weight.kernel = weights_ptr_[26];
        ffn_weights.output_weight.bias = weights_ptr_[27];

        //
        ffn_norm_weights.gamma = weights_ptr_[28];
        ffn_norm_weights.beta = weights_ptr_[29];

        //
        reg_weights.fc1.kernel = weights_ptr_[30];
        reg_weights.fc1.bias = weights_ptr_[31];
        reg_weights.fc2.kernel = weights_ptr_[32];
        reg_weights.fc2.bias = weights_ptr_[33];
        reg_weights.fc3.kernel = weights_ptr_[34];
        reg_weights.fc3.bias = weights_ptr_[35];

        is_maintain_buffer_ = true;
    }

    int embed_dim_;
    int ffn_inter_size_;
    int layer_idx_;

    int num_cams_;
    int num_points_;
    int num_levels_;

    bool is_maintain_buffer_{false};
    T* weights_ptr_[WEIGHT_N]{nullptr};
    size_t weights_size_[WEIGHT_N];
    bool is_maintain_sp_buffer = false;
};

#undef WEIGHT_N

}  // namespace fastertransformer
