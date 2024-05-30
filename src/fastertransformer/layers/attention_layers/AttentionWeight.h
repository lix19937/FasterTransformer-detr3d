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

#pragma once

#include "src/fastertransformer/layers/DenseWeight.h"

namespace fastertransformer {

template<typename T>
struct AttentionWeight {
    DenseWeight<T> query_weight;
    DenseWeight<T> key_weight;
    DenseWeight<T> value_weight;
    DenseWeight<T> attention_output_weight;
};

template<typename T>
struct CAttentionWeight {
    DenseWeight<T> attention_weights;
    DenseWeight<T> output_proj;
    DenseWeight<T> position_encoder_fc1;
    DenseWeight<T> position_encoder_ln1;
    DenseWeight<T> position_encoder_fc2;
    DenseWeight<T> position_encoder_ln2;
};

template<typename T>
struct RegBranchWeight {
    DenseWeight<T> fc1;
    DenseWeight<T> fc2;
    DenseWeight<T> fc3;
};

template<typename T>
struct ClsBranchWeight {
    DenseWeight<T> fc1;
    DenseWeight<T> ln1;
    DenseWeight<T> fc2;
    DenseWeight<T> ln2;
    DenseWeight<T> fc3;
};

}  // namespace fastertransformer
