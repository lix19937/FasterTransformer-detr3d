
WORKSPACE=/home/igs/transformer/FasterTransformer-main
export LD_LIBRARY_PATH=$WORKSPACE/build/lib:${LD_LIBRARY_PATH}


## seq_len = (img_size/patch_size) * (img_size/patch_size) 

#         batch_size img_size patch_size embed_dim head_number with_cls_token data_type int8_mode
./bin/vit_gemm 1    480      16          256      8           1              1          0

#             batch_size img_size patch_size embed_dim head_number layer_num with_cls_token is_fp16
./bin/vit_example 1        480     16         256      8            4        1          1


### refer  to trt config b_16
#         batch_size img_size patch_size embed_dim head_number with_cls_token data_type int8_mode
./bin/vit_gemm 1    480      16          768      12           1              1          0

#             batch_size img_size patch_size embed_dim head_number layer_num with_cls_token is_fp16
./bin/vit_example 1        480     16         768      12            12        1          1