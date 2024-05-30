
WORKSPACE=/home/igs/transformer/FasterTransformer-main

export LD_LIBRARY_PATH=$WORKSPACE/build/lib:${LD_LIBRARY_PATH}
# ldd -r $WORKSPACE/build/lib/libpyt_vit.so
# exit 0  c++filt

cd $WORKSPACE/examples/pytorch/vit

##profile of FP16/FP32 model
python infer_visiontransformer_op.py \
  --model_type=ViT-B_16  \
  --img_size=224 \
  --pretrained_dir=$WORKSPACE/examples/pytorch/vit/ViT-quantization/imagenet21k_ViT-B_16.npz \
  --batch-size=1 \
  --th-path=$WORKSPACE/build/lib/libpyt_vit.so


#
# undefined symbol: _ZN17fastertransformer13cublasAlgoMapC1ESsSs
#  https://github.com/NVIDIA/FasterTransformer/issues/220
#
# [ERROR] CUDA runtime error: CUBLAS_STATUS_NOT_INITIALIZED
# https://github.com/NVIDIA/FasterTransformer/issues/22
#
#
#  multi-head  L=512 can acc    fused{384}  unfused 
#  cross-atten{grid sample, fc, normlize } L=900 vs TRT
#  vit 
#
#       batch_size img_size patch_size embed_dim head_number with_cls_token data_type int8_mode
#  ./bin/vit_gemm 1 224 16 768 12 1 1 0


