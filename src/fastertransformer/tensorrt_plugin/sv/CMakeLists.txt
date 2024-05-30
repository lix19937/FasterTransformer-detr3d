
cmake_minimum_required(VERSION 3.8)

set(sv_trt_files
SVTransformerPlugin.cpp
)

if(BUILD_TRT)
  set(LIB_NAME "svt_plugin")

  add_library(${LIB_NAME} SHARED ${sv_trt_files}) # SHARED
  set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET ${LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE  ON)

  target_link_libraries(${LIB_NAME} SV -lcublas -lcudart -lnvinfer)
endif()
