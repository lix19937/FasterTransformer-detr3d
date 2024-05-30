
from curses import A_TOP
import torch
import torch.nn as nn
import numpy as np
import onnx

def loadtxt_fp32(file_name, shape):
    a = np.loadtxt(file_name, dtype=float)
    a = torch.from_numpy(a)
    a = a.view(shape).float()
    return a

def loadtxt_int(file_name, shape):
    a = np.loadtxt(file_name, dtype=int)
    a = torch.from_numpy(a)
    a = a.view(shape)
    return a

def get_shape(file_name):
    shape_info = file_name.split('.')[-1]
    shape = shape_info.split('-')
    # if len(shape) == 2 and shape[0] == 1:
    #     shape = shape[1]
    return shape

def get_value(file_name):
    shape_info = file_name.split('.')[-1]
    shape = shape_info.split('-')
    new_shape = list(map(int, shape))
   
    return loadtxt_fp32(file_name, new_shape)
    
class SVODTRANSFORMER_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value_0, value_1, value_2, lidar2img, block_1__cross_attention_norm__ln__bias, block_1__cross_attention_norm__ln__weight):
        reg_output = torch.ones([1, 512, 8],dtype=torch.float32).cuda()
        cls_output = torch.ones([1, 512, 8],dtype=torch.float32).cuda()
        return reg_output, cls_output

    @staticmethod
    def symbolic(g, value_0, value_1, value_2, lidar2img, block_1__cross_attention_norm__ln__bias, block_1__cross_attention_norm__ln__weight):
        pc_range_tensor=torch.tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=torch.float32)
        img_shape_tensor=torch.tensor([[288,736]], dtype=torch.int64)

        return g.op("ai.onnx.contrib::SvTransformerDecoder", value_0, value_1, value_2, lidar2img,
          block_1__cross_attention_norm__ln__bias,
          block_1__cross_attention_norm__ln__weight,
          outputs=2,
          max_batch_i=1,
          max_seq_len_i=900,
          seq_len_i=512,
          embed_dim_i=256,
          num_heads_i=8,
          inter_size_i=512,
          layer_num_i=4,
          num_cam_i=6,
          num_reg_points_i=8,
          num_classes_i=8,
          l2i_matr_h_i=4,
          l2i_matr_w_i=4,
          pc_range_t=pc_range_tensor,
          img_shape_t=img_shape_tensor)

 
def SvTransformer(value_0, value_1, value_2, lidar2img, block_1__cross_attention_norm__ln__bias, block_1__cross_attention_norm__ln__weight):
    return SVODTRANSFORMER_.apply(value_0, value_1, value_2, lidar2img, block_1__cross_attention_norm__ln__bias,block_1__cross_attention_norm__ln__weight)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

    def forward(self, value_0, value_1, value_2, lidar2img, block_1__cross_attention_norm__ln__bias, block_1__cross_attention_norm__ln__weight):
        out, reference_points = SvTransformer(value_0, value_1, value_2, lidar2img, block_1__cross_attention_norm__ln__bias, block_1__cross_attention_norm__ln__weight)
        return out, reference_points

def gen_sv_transformer_decoder():
    tfmodel=Model()
    tfmodel.eval()

    value_0 = torch.randn(6, 256, 72, 184).cuda()
    value_1 = torch.randn(6, 256, 36, 92).cuda()
    value_2 = torch.randn(6, 256, 18, 46).cuda()

    lidar2img=torch.randn(6,4,4).cuda()
    
    block_1__cross_attention_norm__ln__bias = torch.randn(1, 256).cuda()
    block_1__cross_attention_norm__ln__weights = torch.randn(1, 256).cuda()

    with torch.no_grad():
        _, _ = tfmodel(value_0,value_1,value_2,lidar2img,block_1__cross_attention_norm__ln__bias,block_1__cross_attention_norm__ln__weights)
    
        torch.onnx.export(tfmodel,
                        (value_0,value_1,value_2, lidar2img, block_1__cross_attention_norm__ln__bias, block_1__cross_attention_norm__ln__weights),
                        'sv_tf_decoder_e2e1008.onnx',
                        input_names=['value_0','value_1','value_2','lidar2img', 'block_1__cross_attention_norm__ln__bias', 'block_1__cross_attention_norm__ln__weights'],
                        output_names=['reg_out','cls_out'],
                        opset_version=11)

def main():        
    gen_sv_transformer_decoder()


if __name__ == '__main__':   
    main()  
    onnx_path='sv_tf_decoder_e2e1008.onnx'
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    values = torch.rand(1, 256)
# https://www.programcreek.com/python/example/122581/onnx.helper.make_tensor
    sub_const_node = onnx.helper.make_tensor(name='block_1__cross_attention_norm__ln__bias',
                          data_type=onnx.TensorProto.FLOAT,
                          dims=np.array(values).shape,
                          vals=values.flatten().tolist())

    values = torch.rand(256, 256)
    sub_const_node2 = onnx.helper.make_tensor(name='block_1__cross_attention_norm__ln__weights',
                          data_type=onnx.TensorProto.FLOAT,
                          dims=np.array(values).shape,
                          vals=values.flatten().tolist())

    graph.initializer.append(sub_const_node)
    graph.initializer.append(sub_const_node2)

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path.replace('1008', '1008_init'))
