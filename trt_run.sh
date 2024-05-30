
WORKSPACE=/home/igs/transformer/FasterTransformer-main
cd $WORKSPACE/examples/tensorrt/vit

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$WORKSPACE/build/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so.11.0:$LD_PRELOAD


##profile of FP16/FP32 model
python3 infer_visiontransformer_plugin.py \
  --model_type=ViT-B_16 \
  --img_size=480 \
  --pretrained_dir=$WORKSPACE/examples/pytorch/vit/ViT-quantization/imagenet21k_ViT-B_16.npz \
  --plugin_path=$WORKSPACE/build/lib/libvit_plugin.so \
  --batch-size=1 \
  --fp16

#       batch_size img_size patch_size embed_dim head_number with_cls_token data_type int8_mode
#  ./bin/vit_gemm 1 224 16 768 12 1 1 0

# nm -Do /usr/local/cuda/lib64/libcudart.so.11.4.108  |grep cudaFree


## seq_len = (img_size/patch_size) * (img_size/patch_size) 
# ./bin/vit_gemm 1 480 16 256 8 1 1 1

