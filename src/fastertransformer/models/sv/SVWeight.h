/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#pragma once

#include "helper_file.h"
#include "helper_macros.h"
#include "src/fastertransformer/layers/DenseWeight.h"
#include "src/fastertransformer/models/sv/SVLayerWeight.h"

namespace fastertransformer {

template <typename T>
struct SVEmbeds {
  const T* query;
  const T* query_pos;
  const T* reference_points;
};

template <typename T>
struct HelperIRPara {
  const T* rp_norm;
  const T* attention_weights_output;
  const T* pos_feat_with_inp_residual;
};

#define WEIGHT_N (16)

template <typename T, typename SRC_T = float> // onnx fp32
struct SVWeight {
  SVWeight() = delete;
  SVWeight(
      const int embed_dim,
      const int inter_size,
      const int num_layer,
      const int seq_len,
      const int num_classes = 5,
      const int num_cams = 12,
      const int num_points = 1,
      const int num_levels = 4,
      const bool with_cls_token = true,
      const bool hold_buffer = true)
      : embed_dim_(embed_dim),
        inter_size_(inter_size),
        num_layer_(num_layer),
        seq_len_(seq_len),
        num_classes_(num_classes),
        num_cams_(num_cams),
        num_points_(num_points),
        num_levels_(num_levels),
        with_cls_token_(with_cls_token) {
    weights_size_[0] = seq_len_ * 1 * embed_dim_; /// query
    weights_size_[1] = seq_len_ * 1 * embed_dim_; /// query_pos
    weights_size_[2] = 1 * seq_len_ * 3; /// reference_points

    // self.num_cams  =12, self.num_points =1, self.num_levels  = 4
    weights_size_[0 + 3] = 1 * 4 * seq_len_; /// !
    weights_size_[1 + 3] = 1 * seq_len_ * num_cams_ * num_points_ * num_levels_;
    weights_size_[2 + 3] = seq_len_ * 1 * embed_dim_;

    // fc
    weights_size_[6] = embed_dim_ * embed_dim_;
    weights_size_[7] = 1 * embed_dim_;
    // ln
    weights_size_[8] = 1 * embed_dim_;
    weights_size_[9] = 1 * embed_dim_;
    // fc
    weights_size_[10] = embed_dim_ * embed_dim_;
    weights_size_[11] = 1 * embed_dim_;
    // ln
    weights_size_[12] = 1 * embed_dim_;
    weights_size_[13] = 1 * embed_dim_;
    // fc
    weights_size_[14] = embed_dim_ * num_classes_;
    weights_size_[15] = 1 * num_classes_;

    if (hold_buffer) {
      for (int i = 0; i < WEIGHT_N; ++i) {
        if (weights_size_[i] == 0) {
          continue;
        }

        deviceMalloc(&weights_ptr_[i], weights_size_[i]);
      }

      setWeightPtr();
    }

    /// --->> transfomer block weights ( alloc mem + pointer bind)
    sv_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
      sv_layer_weights.push_back(
          SVLayerWeight<T, SRC_T>(embed_dim_, inter_size_, i, num_cams_, num_points_, num_levels_, hold_buffer));
    }
  }

  ~SVWeight() {
    if (is_maintain_buffer_) {
      sv_layer_weights.clear();
      for (int i = 0; i < WEIGHT_N; ++i) {
        if (weights_ptr_[i] != nullptr) {
          deviceFree(weights_ptr_[i]);
        }
      }

      pre_transform_embeds.query = nullptr;
      pre_transform_embeds.query_pos = nullptr;
      pre_transform_embeds.reference_points = nullptr;

      helper_irparams.rp_norm = nullptr;
      helper_irparams.attention_weights_output = nullptr;
      helper_irparams.pos_feat_with_inp_residual = nullptr;

      post_transform_weights.fc1.kernel = nullptr;
      post_transform_weights.fc1.bias = nullptr;
      post_transform_weights.ln1.kernel = nullptr;
      post_transform_weights.ln1.bias = nullptr;
      post_transform_weights.fc2.kernel = nullptr;
      post_transform_weights.fc2.bias = nullptr;
      post_transform_weights.ln2.kernel = nullptr;
      post_transform_weights.ln2.bias = nullptr;
      post_transform_weights.fc3.kernel = nullptr;
      post_transform_weights.fc3.bias = nullptr;

      is_maintain_buffer_ = false;
    }
  }

  SVWeight(const SVWeight& other)
      : with_cls_token_(other.with_cls_token_),
        embed_dim_(other.embed_dim_),
        inter_size_(other.inter_size_),
        num_layer_(other.num_layer_),
        seq_len_(other.seq_len_),
        num_classes_(other.num_classes_),
        num_cams_(other.num_cams_),
        num_points_(other.num_points_),
        num_levels_(other.num_levels_)
  {
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

    sv_layer_weights.clear();
    sv_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; ++i) {
      sv_layer_weights.push_back(other.sv_layer_weights[i]);
    }
  }

  SVWeight& operator=(const SVWeight& other) {
    embed_dim_ = other.embed_dim_;
    inter_size_ = other.inter_size_;
    num_layer_ = other.num_layer_;
    seq_len_ = other.seq_len_;
    with_cls_token_ = other.with_cls_token_;
    num_classes_ = other.num_classes_;
    num_cams_ = other.num_cams_;
    num_points_ = other.num_points_;
    num_levels_ = other.num_levels_;

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

    sv_layer_weights.clear();
    sv_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; ++i) {
      sv_layer_weights.push_back(other.sv_layer_weights[i]);
    }

    return *this;
  }

  size_t GetSerializeSize() {
    size_t count{0};
    for (int i = 0; i < WEIGHT_N; ++i) {
      count += weights_size_[i];
    }

    if (std::is_same<SRC_T, float>::value) {
      count *= sizeof(SRC_T);
    } else {
      count *= sizeof(T);
    }

    for (auto& lw : sv_layer_weights) {
      count += lw.GetSerializeSize();
    }

    return count;
  }

  size_t getworkspace() const {
    size_t max_sz = 0;
    for (int i = 0; i < WEIGHT_N; ++i) {
      max_sz = std::max(max_sz, weights_size_[i]);
    }
    return max_sz;
  }

  void Hfp32ToDhalf(const float* w, half* h_ptr, const void* d_ptr, const int idx) {
    for (size_t i = 0; i < weights_size_[idx]; ++i) {
      h_ptr[i] = w[i];
    }

    cudaMemcpy((half*)d_ptr, h_ptr, sizeof(half) * weights_size_[idx], cudaMemcpyHostToDevice);
  }

  void DhalfToHfp32(half* h_ptr, float* ptr, const int idx) {
    cudaMemcpy(h_ptr, weights_ptr_[idx], sizeof(half) * weights_size_[idx], cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < weights_size_[idx]; ++i) {
      ptr[i] = h_ptr[i];
    }
  }

  void serialize(void* buffer) {
    char* tmp_buf = (char*)buffer;

    if (std::is_same<T, half>::value && std::is_same<SRC_T, float>::value) {
      T* buf = (T*)malloc(getworkspace() * sizeof(T));
      for (int i = 0; i < WEIGHT_N; ++i) {
        DhalfToHfp32((half*)buf, (float*)tmp_buf, i);
        tmp_buf += sizeof(SRC_T) * weights_size_[i];
      }
      free(buf);

      for (auto& lw : sv_layer_weights) {
        lw.serialize(tmp_buf);
        tmp_buf += lw.GetSerializeSize();
      }
      return;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    for (int i = 0; i < WEIGHT_N; ++i) {
      cudaMemcpy(tmp_buf, weights_ptr_[i], sizeof(T) * weights_size_[i], cudaMemcpyDeviceToHost);
      tmp_buf += sizeof(T) * weights_size_[i];
    }

    for (auto& lw : sv_layer_weights) {
      lw.serialize(tmp_buf);
      tmp_buf += lw.GetSerializeSize();
    }
  }

  void deserialize(const void* buffer) {
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

      for (auto& lw : sv_layer_weights) {
        lw.deserialize(tmp_buf);
        tmp_buf += lw.GetSerializeSize();
      }
      return;
    }

    for (int i = 0; i < WEIGHT_N; ++i) {
      cudaMemcpy(weights_ptr_[i], tmp_buf, sizeof(T) * weights_size_[i], cudaMemcpyHostToDevice);
      tmp_buf += sizeof(T) * weights_size_[i];
    }

    for (auto& lw : sv_layer_weights) {
      lw.deserialize(tmp_buf);
      tmp_buf += lw.GetSerializeSize();
    }
  }

  void CopyWeightsFromHostBuffersFp32ToDeviceHalf(const float* const*& w) {
    half* h_ptr = (half*)malloc(getworkspace() * sizeof(half));
    int i = 0;

    Hfp32ToDhalf(*w++, h_ptr, pre_transform_embeds.query, i++);
    Hfp32ToDhalf(*w++, h_ptr, pre_transform_embeds.query_pos, i++);
    Hfp32ToDhalf(*w++, h_ptr, pre_transform_embeds.reference_points, i++);

    Hfp32ToDhalf(*w++, h_ptr, helper_irparams.rp_norm, i++);
    Hfp32ToDhalf(*w++, h_ptr, helper_irparams.attention_weights_output, i++);
    Hfp32ToDhalf(*w++, h_ptr, helper_irparams.pos_feat_with_inp_residual, i++);

    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.fc1.kernel, i++);
    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.fc1.bias, i++);
    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.ln1.kernel, i++);
    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.ln1.bias, i++);
    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.fc2.kernel, i++);
    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.fc2.bias, i++);
    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.ln2.kernel, i++);
    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.ln2.bias, i++);
    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.fc3.kernel, i++);
    Hfp32ToDhalf(*w++, h_ptr, post_transform_weights.fc3.bias, i);

    // ExportWeights();
    free(h_ptr);

    for (i = 0; i < num_layer_; ++i) {
      auto& layer_weight = sv_layer_weights[i];
      layer_weight.CopyWeightsFromHostBuffersFp32ToDeviceHalf(w);

      // layer_weight.ExportWeights(i);
    }
  }

  void CopyWeightsFromHostBuffers(const T* const*& w) {
    // clang-format off
        if (with_cls_token_) {
            cudaMemcpy(const_cast<T*>(pre_transform_embeds.query),            *w++, sizeof(T) * weights_size_[0],cudaMemcpyHostToDevice);
            cudaMemcpy(const_cast<T*>(pre_transform_embeds.query_pos),        *w++, sizeof(T) * weights_size_[1],cudaMemcpyHostToDevice);
            cudaMemcpy(const_cast<T*>(pre_transform_embeds.reference_points), *w++, sizeof(T) * weights_size_[2],cudaMemcpyHostToDevice);
        }

        /// helper ir
        cudaMemcpy(const_cast<T*>(helper_irparams.rp_norm),                   *w++, sizeof(T) * weights_size_[3],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(helper_irparams.attention_weights_output),  *w++, sizeof(T) * weights_size_[4],cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(helper_irparams.pos_feat_with_inp_residual),*w++, sizeof(T) * weights_size_[5],cudaMemcpyHostToDevice);

        /// post 
        cudaMemcpy(const_cast<T*>(post_transform_weights.fc1.kernel), *w++, sizeof(T) * weights_size_[6], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transform_weights.fc1.bias  ), *w++, sizeof(T) * weights_size_[7], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transform_weights.ln1.kernel), *w++, sizeof(T) * weights_size_[8], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transform_weights.ln1.bias  ), *w++, sizeof(T) * weights_size_[9], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transform_weights.fc2.kernel), *w++, sizeof(T) * weights_size_[10], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transform_weights.fc2.bias  ), *w++, sizeof(T) * weights_size_[11], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transform_weights.ln2.kernel), *w++, sizeof(T) * weights_size_[12], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transform_weights.ln2.bias  ), *w++, sizeof(T) * weights_size_[13], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transform_weights.fc3.kernel), *w++, sizeof(T) * weights_size_[14], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transform_weights.fc3.bias  ), *w++, sizeof(T) * weights_size_[15], cudaMemcpyHostToDevice);

        /// block
        for (int i = 0; i < num_layer_; ++i) {
            auto& layer_weight = sv_layer_weights[i];
            layer_weight.CopyWeightsFromHostBuffers(w);
        }
    // clang-format on
  }

  inline size_t GetWeightCount() {
    size_t weight_count = WEIGHT_N;
    weight_count += num_layer_ * sv_layer_weights[0].GetWeightCount();

    return weight_count;
  }

  void ExportWeights() const {
    int i = 0;
    // clang-format off
    cudaacc::WriteFromDptr("./from_plugin/pre/posembed.in.query.512-1-256", {512, 1, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/pre/posembed.in.query_pos.512-1-256", {512, 1, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/pre/reg.in.reference_points.1-512-3", {1, 512, 3}, weights_ptr_[i++]);

    cudaacc::WriteFromDptr("./from_plugin/ir/ir.ca.fs.rfpcat.1-4-512", {1, 4, 512}, weights_ptr_[i++]);
    const int __dims3 = num_cams_ * num_points_ * num_levels_;
    cudaacc::WriteFromDptr("./from_plugin/ir/ir.ca.attention_weights.out_nobias.1-512-" + std::to_string(__dims3), {1, 512, __dims3}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/ir/ir.ca.out.inp_res_pos_feat.512-1-256", {512, 1, 256}, weights_ptr_[i++]);

    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.fc1.weight.256-256", {256, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.fc1.bias.1-256", {1, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.ln1.weight.1-256", {1, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.ln1.bias.1-256", {1, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.fc2.weight.256-256", {256, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.fc2.bias.1-256", {1, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.ln2.weight.1-256", {1, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.ln2.bias.1-256", {1, 256}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.fc3.weight.256-5", {256, 5}, weights_ptr_[i++]);
    cudaacc::WriteFromDptr("./from_plugin/post/cls_branches.fc3.bias.1-5",{1, 5}, weights_ptr_[i]);
    // clang-format on

    for (i = 0; i < num_layer_; ++i) {
      auto& layer_weight = sv_layer_weights[i];
      layer_weight.ExportWeights(i);
    }
  }

  SVEmbeds<T> pre_transform_embeds;
  HelperIRPara<T> helper_irparams;
  std::vector<SVLayerWeight<T, SRC_T>> sv_layer_weights;
  ClsBranchWeight<T> post_transform_weights;

  T*& GetPtr(const int i) { return weights_ptr_[i]; }

 private:
  void setWeightPtr() {
    int i = 0;
    pre_transform_embeds.query = weights_ptr_[i++];
    pre_transform_embeds.query_pos = weights_ptr_[i++];
    pre_transform_embeds.reference_points = weights_ptr_[i++];

    helper_irparams.rp_norm = weights_ptr_[i++];
    helper_irparams.attention_weights_output = weights_ptr_[i++];
    helper_irparams.pos_feat_with_inp_residual = weights_ptr_[i++];

    post_transform_weights.fc1.kernel = weights_ptr_[i++];
    post_transform_weights.fc1.bias = weights_ptr_[i++];
    post_transform_weights.ln1.kernel = weights_ptr_[i++];
    post_transform_weights.ln1.bias = weights_ptr_[i++];
    post_transform_weights.fc2.kernel = weights_ptr_[i++];
    post_transform_weights.fc2.bias = weights_ptr_[i++];
    post_transform_weights.ln2.kernel = weights_ptr_[i++];
    post_transform_weights.ln2.bias = weights_ptr_[i++];
    post_transform_weights.fc3.kernel = weights_ptr_[i++];
    post_transform_weights.fc3.bias = weights_ptr_[i];

    is_maintain_buffer_ = true;
  }

  int embed_dim_;
  int inter_size_;
  int num_layer_;
  int seq_len_;
  int num_classes_;

  int num_cams_;
  int num_points_;
  int num_levels_;

  bool with_cls_token_;

  bool is_maintain_buffer_{false};
  T* weights_ptr_[WEIGHT_N]{nullptr};
  size_t weights_size_[WEIGHT_N];
};

#undef WEIGHT_N
} // namespace fastertransformer
