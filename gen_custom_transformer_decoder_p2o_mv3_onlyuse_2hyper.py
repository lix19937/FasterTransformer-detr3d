
from curses import A_TOP
import torch
import torch.nn as nn
import numpy as np

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
    def forward(ctx, value_0, value_1, value_2, lidar2img):
        reg_output = torch.ones([1, 512, 8],dtype=torch.float32).cuda()
        cls_output = torch.ones([1, 512, 8],dtype=torch.float32).cuda()
        return reg_output, cls_output

    @staticmethod
    def symbolic(g, value_0, value_1, value_2, lidar2img):
        pc_range_tensor=torch.tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=torch.float32)
        img_shape_tensor=torch.tensor([[288,736]], dtype=torch.int64)
        
        return g.op("ai.onnx.contrib::SvTransformerDecoder", value_0, value_1, value_2, lidar2img,  
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
          img_shape_t=img_shape_tensor,
          block_1__cross_attention__attention_weights__fc__bias_t=cross_attention__attention_weights__fc__bias_list[0],
          block_1__cross_attention__attention_weights__fc__weight_t=cross_attention__attention_weights__fc__weight_list[0]       
          )

 
def SvTransformer(value_0, value_1, value_2, lidar2img):
    return SVODTRANSFORMER_.apply(value_0, value_1, value_2, lidar2img)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

    def forward(self, value_0, value_1, value_2, lidar2img):
        out, reference_points = SvTransformer(value_0, value_1, value_2, lidar2img)
        return out, reference_points

def gen_sv_transformer_decoder():
    tfmodel=Model()
    tfmodel.eval()

    value_0 = torch.randn(6, 256, 72, 184).cuda()
    value_1 = torch.randn(6, 256, 36, 92).cuda()
    value_2 = torch.randn(6, 256, 18, 46).cuda()


    lidar2img=torch.randn(6,4,4).cuda()
    
    with torch.no_grad():
        _, _ = tfmodel(value_0,value_1,value_2,lidar2img)
    
        torch.onnx.export(tfmodel,
                        (value_0,value_1,value_2, lidar2img),
                        'sv_tf_decoder_e2e1018.onnx',
                        input_names=['value_0','value_1','value_2','lidar2img'],
                        output_names=['reg_out','cls_out'],
                        opset_version=11)

def main():
    root_dir = "./tf_decoder_weights/"

    ###
    for lid in range(1):
        blockid = lid + 1

        block_1__cross_attention__attention_weights__fc__bias_file = root_dir + "block_"+str(blockid)+".cross_attention.attention_weights.fc.bias.1-24"
        block_1__cross_attention__attention_weights__fc__bias = get_value(block_1__cross_attention__attention_weights__fc__bias_file)
        cross_attention__attention_weights__fc__bias_list.append(block_1__cross_attention__attention_weights__fc__bias)
        
        block_1__cross_attention__attention_weights__fc__weight_file = root_dir + "block_"+str(blockid)+".cross_attention.attention_weights.fc.weight.256-24"
        block_1__cross_attention__attention_weights__fc__weight = get_value(block_1__cross_attention__attention_weights__fc__weight_file)
        cross_attention__attention_weights__fc__weight_list.append(block_1__cross_attention__attention_weights__fc__weight)              
    
    gen_sv_transformer_decoder()


if __name__ == '__main__':
    cross_attention__attention_weights__fc__bias_list=[]
    cross_attention__attention_weights__fc__weight_list=[]
    
    main()  
