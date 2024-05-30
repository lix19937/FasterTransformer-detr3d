# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.8)

add_executable(sv_example_fp16_linear   sv_example_fp16_linear.cc)
target_link_libraries(sv_example_fp16_linear PUBLIC SV vit_kernels
  cublasMMWrapper -lcublas -lcublasLt -lcudart  -lm)

add_executable(sv_example_fp32_linear   sv_example_fp32_linear.cc)
target_link_libraries(sv_example_fp32_linear PUBLIC SV vit_kernels
  cublasMMWrapper -lcublas -lcublasLt -lcudart  -lm)

add_executable(sv_example_int8_chw32   sv_example_int8_chw32.cc)
target_link_libraries(sv_example_int8_chw32 PUBLIC SV vit_kernels
  cublasMMWrapper -lcublas -lcublasLt -lcudart  -lm)

  