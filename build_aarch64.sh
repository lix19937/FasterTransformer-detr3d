
cd build
rm -fr ../build/c*  ../build/C*

cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/cmake_aarch64.toolchain \
-DCUDA_VERSION=11.4 -DNDEBUG=ON -DSM=87 -DCMAKE_BUILD_TYPE=Release -DBUILD_TRT=ON .. && make  -j44


# -DCUDNN_LIB=/pdk_files/cudnn/usr/lib/aarch64-linux-gnu/libcudnn.so 
# -DCUBLAS_LIB=/usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/libcublas.so 
# -DCUBLASLT_LIB=/usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/libcublasLt.so 

# ./bin/vit_gemm 1 384 16 768 12 1 1 0
# ./bin/vit_example 1 384 16 768 12 12 1 1

## seq_len = (img_size/patch_size) * (img_size/patch_size) 
# ./bin/vit_gemm 1 480 16 256 8 1 1 0
# ./bin/vit_example 1 480 16 256 8 4 1 1