
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
    # if len(new_shape) == 2 and new_shape[0] == 1:
    #     new_shape = new_shape[1]
    
    return loadtxt_fp32(file_name, new_shape)
    
class SVODTRANSFORMER_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value_0, value_1, value_2, lidar2img, img_shape):
        reg_output = torch.ones([1, 512, 8],dtype=torch.float32).cuda()
        cls_output = torch.ones([1, 512, 5],dtype=torch.float32).cuda()
        return reg_output, cls_output

    @staticmethod
    def symbolic(g, value_0, value_1, value_2, lidar2img, img_shape):
        pc_range_tensor=torch.tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=torch.float32)
        
        return g.op("ai.onnx.contrib::SvTransformerDecoder", value_0, value_1, value_2, lidar2img,  img_shape,
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
          num_classes_i=5,
          l2i_matr_h_i=4,
          l2i_matr_w_i=4,
          pc_range_t=pc_range_tensor,

          posembed__in__query_t=posembed__in__query,
          posembed__in__query_pos_t=posembed__in__query_pos,
          reg__in__reference_points_t=reg__in__reference_points,

          ir__ca__fs__rfpcat_t=ir__ca__fs__rfpcat, 
          ir__ca__attention_weights__out_nobias_t=ir__ca__attention_weights__out_nobias, 
          ir__ca__out__inp_res_pos_feat_t=ir__ca__out__inp_res_pos_feat,

          cls_branches__fc1__weight_t = cls_branches__fc1__weight,
          cls_branches__fc1__bias_t   = cls_branches__fc1__bias,
          cls_branches__ln1__weight_t = cls_branches__ln1__weight,
          cls_branches__ln1__bias_t   = cls_branches__ln1__bias,
          cls_branches__fc2__weight_t = cls_branches__fc2__weight,
          cls_branches__fc2__bias_t   = cls_branches__fc2__bias,
          cls_branches__ln2__weight_t = cls_branches__ln2__weight,
          cls_branches__ln2__bias_t   = cls_branches__ln2__bias,
          cls_branches__fc3__weight_t = cls_branches__fc3__weight,
          cls_branches__fc3__bias_t   = cls_branches__fc3__bias,

          block_1__cross_attention__attention_weights__fc__bias_t=cross_attention__attention_weights__fc__bias_list[0],
          block_1__cross_attention__attention_weights__fc__weight_t=cross_attention__attention_weights__fc__weight_list[0],
          block_1__cross_attention_norm__ln__bias_t=cross_attention_norm__ln__bias_list[0],
          block_1__cross_attention_norm__ln__weight_t=cross_attention_norm__ln__weight_list[0],
          block_1__cross_attention__output_proj__fc__bias_t=cross_attention__output_proj__fc__bias_list[0],
          block_1__cross_attention__output_proj__fc__weight_t=cross_attention__output_proj__fc__weight_list[0],
          block_1__cross_attention__position_encoder__fc1__bias_t=cross_attention__position_encoder__fc1__bias_list[0],
          block_1__cross_attention__position_encoder__fc1__weight_t=cross_attention__position_encoder__fc1__weight_list[0],
          block_1__cross_attention__position_encoder__fc2__bias_t=cross_attention__position_encoder__fc2__bias_list[0],
          block_1__cross_attention__position_encoder__fc2__weight_t=cross_attention__position_encoder__fc2__weight_list[0],
          block_1__cross_attention__position_encoder__ln1__bias_t=cross_attention__position_encoder__ln1__bias_list[0],
          block_1__cross_attention__position_encoder__ln1__weight_t=cross_attention__position_encoder__ln1__weight_list[0],
          block_1__cross_attention__position_encoder__ln2__bias_t=cross_attention__position_encoder__ln2__bias_list[0],
          block_1__cross_attention__position_encoder__ln2__weight_t=cross_attention__position_encoder__ln2__weight_list[0],
          block_1__ffn__fc1__bias_t=ffn__fc1__bias_list[0],
          block_1__ffn__fc1__weight_t=ffn__fc1__weight_list[0],
          block_1__ffn__fc2__bias_t=ffn__fc2__bias_list[0],
          block_1__ffn__fc2__weight_t=ffn__fc2__weight_list[0],
          block_1__ffn_norm__ln__bias_t=ffn_norm__ln__bias_list[0],
          block_1__ffn_norm__ln__weight_t=ffn_norm__ln__weight_list[0],
          block_1__mh_attention__key__bias_t=mh_attention__key__bias_list[0],
          block_1__mh_attention__key__weight_t=mh_attention__key__weight_list[0],
          block_1__mh_attention__query__bias_t=mh_attention__query__bias_list[0],
          block_1__mh_attention__query__weight_t=mh_attention__query__weight_list[0],
          block_1__mh_attention__value__bias_t=mh_attention__value__bias_list[0],
          block_1__mh_attention__value__weight_t=mh_attention__value__weight_list[0],
          block_1__mh_attention__out__bias_t=mh_attention__out__bias_list[0],
          block_1__mh_attention__out__weight_t=mh_attention__out__weight_list[0],
          block_1__mh_attention_norm__ln__bias_t=mh_attention_norm__ln__bias_list[0],
          block_1__mh_attention_norm__ln__weight_t=mh_attention_norm__ln__weight_list[0],
          block_1__reg_branches__fc1__bias_t=reg_branches__fc1__bias_list[0],
          block_1__reg_branches__fc1__weight_t=reg_branches__fc1__weight_list[0],
          block_1__reg_branches__fc2__bias_t=reg_branches__fc2__bias_list[0],
          block_1__reg_branches__fc2__weight_t=reg_branches__fc2__weight_list[0],
          block_1__reg_branches__fc3__bias_t=reg_branches__fc3__bias_list[0],
          block_1__reg_branches__fc3__weight_t=reg_branches__fc3__weight_list[0],
          block_2__cross_attention__attention_weights__fc__bias_t=cross_attention__attention_weights__fc__bias_list[1],
          block_2__cross_attention__attention_weights__fc__weight_t=cross_attention__attention_weights__fc__weight_list[1],
          block_2__cross_attention_norm__ln__bias_t=cross_attention_norm__ln__bias_list[1],
          block_2__cross_attention_norm__ln__weight_t=cross_attention_norm__ln__weight_list[1],
          block_2__cross_attention__output_proj__fc__bias_t=cross_attention__output_proj__fc__bias_list[1],
          block_2__cross_attention__output_proj__fc__weight_t=cross_attention__output_proj__fc__weight_list[1],
          block_2__cross_attention__position_encoder__fc1__bias_t=cross_attention__position_encoder__fc1__bias_list[1],
          block_2__cross_attention__position_encoder__fc1__weight_t=cross_attention__position_encoder__fc1__weight_list[1],
          block_2__cross_attention__position_encoder__fc2__bias_t=cross_attention__position_encoder__fc2__bias_list[1],
          block_2__cross_attention__position_encoder__fc2__weight_t=cross_attention__position_encoder__fc2__weight_list[1],
          block_2__cross_attention__position_encoder__ln1__bias_t=cross_attention__position_encoder__ln1__bias_list[1],
          block_2__cross_attention__position_encoder__ln1__weight_t=cross_attention__position_encoder__ln1__weight_list[1],
          block_2__cross_attention__position_encoder__ln2__bias_t=cross_attention__position_encoder__ln2__bias_list[1],
          block_2__cross_attention__position_encoder__ln2__weight_t=cross_attention__position_encoder__ln2__weight_list[1],
          block_2__ffn__fc1__bias_t=ffn__fc1__bias_list[1],
          block_2__ffn__fc1__weight_t=ffn__fc1__weight_list[1],
          block_2__ffn__fc2__bias_t=ffn__fc2__bias_list[1],
          block_2__ffn__fc2__weight_t=ffn__fc2__weight_list[1],
          block_2__ffn_norm__ln__bias_t=ffn_norm__ln__bias_list[1],
          block_2__ffn_norm__ln__weight_t=ffn_norm__ln__weight_list[1],
          block_2__mh_attention__key__bias_t=mh_attention__key__bias_list[1],
          block_2__mh_attention__key__weight_t=mh_attention__key__weight_list[1],
          block_2__mh_attention__query__bias_t=mh_attention__query__bias_list[1],
          block_2__mh_attention__query__weight_t=mh_attention__query__weight_list[1],
          block_2__mh_attention__value__bias_t=mh_attention__value__bias_list[1],
          block_2__mh_attention__value__weight_t=mh_attention__value__weight_list[1],
          block_2__mh_attention__out__bias_t=mh_attention__out__bias_list[1],
          block_2__mh_attention__out__weight_t=mh_attention__out__weight_list[1],
          block_2__mh_attention_norm__ln__bias_t=mh_attention_norm__ln__bias_list[1],
          block_2__mh_attention_norm__ln__weight_t=mh_attention_norm__ln__weight_list[1],
          block_2__reg_branches__fc1__bias_t=reg_branches__fc1__bias_list[1],
          block_2__reg_branches__fc1__weight_t=reg_branches__fc1__weight_list[1],
          block_2__reg_branches__fc2__bias_t=reg_branches__fc2__bias_list[1],
          block_2__reg_branches__fc2__weight_t=reg_branches__fc2__weight_list[1],
          block_2__reg_branches__fc3__bias_t=reg_branches__fc3__bias_list[1],
          block_2__reg_branches__fc3__weight_t=reg_branches__fc3__weight_list[1],
          block_3__cross_attention__attention_weights__fc__bias_t=cross_attention__attention_weights__fc__bias_list[2],
          block_3__cross_attention__attention_weights__fc__weight_t=cross_attention__attention_weights__fc__weight_list[2],
          block_3__cross_attention_norm__ln__bias_t=cross_attention_norm__ln__bias_list[2],
          block_3__cross_attention_norm__ln__weight_t=cross_attention_norm__ln__weight_list[2],
          block_3__cross_attention__output_proj__fc__bias_t=cross_attention__output_proj__fc__bias_list[2],
          block_3__cross_attention__output_proj__fc__weight_t=cross_attention__output_proj__fc__weight_list[2],
          block_3__cross_attention__position_encoder__fc1__bias_t=cross_attention__position_encoder__fc1__bias_list[2],
          block_3__cross_attention__position_encoder__fc1__weight_t=cross_attention__position_encoder__fc1__weight_list[2],
          block_3__cross_attention__position_encoder__fc2__bias_t=cross_attention__position_encoder__fc2__bias_list[2],
          block_3__cross_attention__position_encoder__fc2__weight_t=cross_attention__position_encoder__fc2__weight_list[2],
          block_3__cross_attention__position_encoder__ln1__bias_t=cross_attention__position_encoder__ln1__bias_list[2],
          block_3__cross_attention__position_encoder__ln1__weight_t=cross_attention__position_encoder__ln1__weight_list[2],
          block_3__cross_attention__position_encoder__ln2__bias_t=cross_attention__position_encoder__ln2__bias_list[2],
          block_3__cross_attention__position_encoder__ln2__weight_t=cross_attention__position_encoder__ln2__weight_list[2],
          block_3__ffn__fc1__bias_t=ffn__fc1__bias_list[2],
          block_3__ffn__fc1__weight_t=ffn__fc1__weight_list[2],
          block_3__ffn__fc2__bias_t=ffn__fc2__bias_list[2],
          block_3__ffn__fc2__weight_t=ffn__fc2__weight_list[2],
          block_3__ffn_norm__ln__bias_t=ffn_norm__ln__bias_list[2],
          block_3__ffn_norm__ln__weight_t=ffn_norm__ln__weight_list[2],
          block_3__mh_attention__key__bias_t=mh_attention__key__bias_list[2],
          block_3__mh_attention__key__weight_t=mh_attention__key__weight_list[2],
          block_3__mh_attention__query__bias_t=mh_attention__query__bias_list[2],
          block_3__mh_attention__query__weight_t=mh_attention__query__weight_list[2],
          block_3__mh_attention__value__bias_t=mh_attention__value__bias_list[2],
          block_3__mh_attention__value__weight_t=mh_attention__value__weight_list[2],
          block_3__mh_attention__out__bias_t=mh_attention__out__bias_list[2],
          block_3__mh_attention__out__weight_t=mh_attention__out__weight_list[2],
          block_3__mh_attention_norm__ln__bias_t=mh_attention_norm__ln__bias_list[2],
          block_3__mh_attention_norm__ln__weight_t=mh_attention_norm__ln__weight_list[2],
          block_3__reg_branches__fc1__bias_t=reg_branches__fc1__bias_list[2],
          block_3__reg_branches__fc1__weight_t=reg_branches__fc1__weight_list[2],
          block_3__reg_branches__fc2__bias_t=reg_branches__fc2__bias_list[2],
          block_3__reg_branches__fc2__weight_t=reg_branches__fc2__weight_list[2],
          block_3__reg_branches__fc3__bias_t=reg_branches__fc3__bias_list[2],
          block_3__reg_branches__fc3__weight_t=reg_branches__fc3__weight_list[2],
          block_4__cross_attention__attention_weights__fc__bias_t=cross_attention__attention_weights__fc__bias_list[3],
          block_4__cross_attention__attention_weights__fc__weight_t=cross_attention__attention_weights__fc__weight_list[3],
          block_4__cross_attention_norm__ln__bias_t=cross_attention_norm__ln__bias_list[3],
          block_4__cross_attention_norm__ln__weight_t=cross_attention_norm__ln__weight_list[3],
          block_4__cross_attention__output_proj__fc__bias_t=cross_attention__output_proj__fc__bias_list[3],
          block_4__cross_attention__output_proj__fc__weight_t=cross_attention__output_proj__fc__weight_list[3],
          block_4__cross_attention__position_encoder__fc1__bias_t=cross_attention__position_encoder__fc1__bias_list[3],
          block_4__cross_attention__position_encoder__fc1__weight_t=cross_attention__position_encoder__fc1__weight_list[3],
          block_4__cross_attention__position_encoder__fc2__bias_t=cross_attention__position_encoder__fc2__bias_list[3],
          block_4__cross_attention__position_encoder__fc2__weight_t=cross_attention__position_encoder__fc2__weight_list[3],
          block_4__cross_attention__position_encoder__ln1__bias_t=cross_attention__position_encoder__ln1__bias_list[3],
          block_4__cross_attention__position_encoder__ln1__weight_t=cross_attention__position_encoder__ln1__weight_list[3],
          block_4__cross_attention__position_encoder__ln2__bias_t=cross_attention__position_encoder__ln2__bias_list[3],
          block_4__cross_attention__position_encoder__ln2__weight_t=cross_attention__position_encoder__ln2__weight_list[3],
          block_4__ffn__fc1__bias_t=ffn__fc1__bias_list[3],
          block_4__ffn__fc1__weight_t=ffn__fc1__weight_list[3],
          block_4__ffn__fc2__bias_t=ffn__fc2__bias_list[3],
          block_4__ffn__fc2__weight_t=ffn__fc2__weight_list[3],
          block_4__ffn_norm__ln__bias_t=ffn_norm__ln__bias_list[3],
          block_4__ffn_norm__ln__weight_t=ffn_norm__ln__weight_list[3],
          block_4__mh_attention__key__bias_t=mh_attention__key__bias_list[3],
          block_4__mh_attention__key__weight_t=mh_attention__key__weight_list[3],
          block_4__mh_attention__query__bias_t=mh_attention__query__bias_list[3],
          block_4__mh_attention__query__weight_t=mh_attention__query__weight_list[3],
          block_4__mh_attention__value__bias_t=mh_attention__value__bias_list[3],
          block_4__mh_attention__value__weight_t=mh_attention__value__weight_list[3],
          block_4__mh_attention__out__bias_t=mh_attention__out__bias_list[3],
          block_4__mh_attention__out__weight_t=mh_attention__out__weight_list[3],
          block_4__mh_attention_norm__ln__bias_t=mh_attention_norm__ln__bias_list[3],
          block_4__mh_attention_norm__ln__weight_t=mh_attention_norm__ln__weight_list[3],
          block_4__reg_branches__fc1__bias_t=reg_branches__fc1__bias_list[3],
          block_4__reg_branches__fc1__weight_t=reg_branches__fc1__weight_list[3],
          block_4__reg_branches__fc2__bias_t=reg_branches__fc2__bias_list[3],
          block_4__reg_branches__fc2__weight_t=reg_branches__fc2__weight_list[3],
          block_4__reg_branches__fc3__bias_t=reg_branches__fc3__bias_list[3],
          block_4__reg_branches__fc3__weight_t=reg_branches__fc3__weight_list[3]
          )

 
def SvTransformer(value_0, value_1, value_2, lidar2img,img_shape):
    return SVODTRANSFORMER_.apply(value_0, value_1, value_2, lidar2img,img_shape)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.Conv_282 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True).cuda()
        self.Conv_283 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True).cuda()
        self.Conv_285 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True).cuda()
        self.Conv_289 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True).cuda()
        self.Conv_284 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True).cuda()
        self.Conv_287 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True).cuda()
        self.Conv_290 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True).cuda()
        
    def forward(self, x1, x2, x3, lidar2img):
        value_0 = self.Conv_282(x1)
        value_1_t = self.Conv_283(x2) + self.Conv_285(value_0)
        value_1 = self.Conv_289(value_1_t)
        value_2_t = self.Conv_284(x3) + self.Conv_287(value_1_t)
        value_2 = self.Conv_290(value_2_t)
        img_shape = img_shape=torch.tensor([[192,736]]).cuda()

        out, reference_points = SvTransformer(value_0, value_1, value_2, lidar2img,img_shape)
        return out, reference_points

def gen_sv_transformer_decoder():
    tfmodel=Model()
    tfmodel.eval()

    value_0 = torch.randn(6, 256, 48, 184).cuda()
    value_1 = torch.randn(6, 256, 24, 92).cuda()
    value_2 = torch.randn(6, 256, 12, 46).cuda()

    lidar2img=torch.randn(6,4,4).cuda()
    
    with torch.no_grad():
        _, _ = tfmodel(value_0,value_1,value_2,lidar2img)
    
        torch.onnx.export(tfmodel,
                        (value_0,value_1,value_2,lidar2img),
                        'sv_tf_decoder_e2e_with_conv.onnx',
                        input_names=['value_0','value_1','value_2','lidar2img'],
                        output_names=['reg_out','cls_out'],
                        opset_version=11)

def main():
    global posembed__in__query
    global posembed__in__query_pos
    global reg__in__reference_points
    
    global ir__ca__fs__rfpcat 
    global ir__ca__attention_weights__out_nobias 
    global ir__ca__out__inp_res_pos_feat 

    global cls_branches__fc1__weight 
    global cls_branches__fc1__bias   
    global cls_branches__ln1__weight 
    global cls_branches__ln1__bias   
    global cls_branches__fc2__weight 
    global cls_branches__fc2__bias   
    global cls_branches__ln2__weight 
    global cls_branches__ln2__bias   
    global cls_branches__fc3__weight 
    global cls_branches__fc3__bias

    root_dir = "./tf_decoder_weights/"

    ###  
    pre_weight_name = ["posembed.in.query.512-1-256",
                       "posembed.in.query_pos.512-1-256",
                       "reg.in.reference_points.1-512-3"]

    posembed__in__query       = get_value(root_dir + "pre/" + pre_weight_name[0])
    posembed__in__query_pos   = get_value(root_dir + "pre/" + pre_weight_name[1])
    reg__in__reference_points = get_value(root_dir + "pre/" + pre_weight_name[2])

    ###
    helper_irparam_names = ["ir.ca.fs.rfpcat.1-4-512",
                            "ir.ca.attention_weights.out_nobias.1-512-24",
                            "ir.ca.out.inp_res_pos_feat.512-1-256"]
    ir__ca__fs__rfpcat                    = get_value(root_dir + "ir/" + helper_irparam_names[0])
    ir__ca__attention_weights__out_nobias = get_value(root_dir + "ir/" + helper_irparam_names[1])
    ir__ca__out__inp_res_pos_feat         = get_value(root_dir + "ir/" + helper_irparam_names[2])

    ###
    post_weight_names = [ "cls_branches.fc1.weight.256-256",
                          "cls_branches.fc1.bias.1-256", 
                          "cls_branches.ln1.weight.1-256", 
                          "cls_branches.ln1.bias.1-256", 
                          "cls_branches.fc2.weight.256-256", 
                          "cls_branches.fc2.bias.1-256", 
                          "cls_branches.ln2.weight.1-256", 
                          "cls_branches.ln2.bias.1-256", 
                          "cls_branches.fc3.weight.256-5", 
                          "cls_branches.fc3.bias.1-5"]    
    cls_branches__fc1__weight = get_value(root_dir + "post/" + post_weight_names[0])
    cls_branches__fc1__bias   = get_value(root_dir + "post/" + post_weight_names[1])
    cls_branches__ln1__weight = get_value(root_dir + "post/" + post_weight_names[2])
    cls_branches__ln1__bias   = get_value(root_dir + "post/" + post_weight_names[3])
    cls_branches__fc2__weight = get_value(root_dir + "post/" + post_weight_names[4])
    cls_branches__fc2__bias   = get_value(root_dir + "post/" + post_weight_names[5])
    cls_branches__ln2__weight = get_value(root_dir + "post/" + post_weight_names[6])
    cls_branches__ln2__bias   = get_value(root_dir + "post/" + post_weight_names[7]) 
    cls_branches__fc3__weight = get_value(root_dir + "post/" + post_weight_names[8]) 
    cls_branches__fc3__bias   = get_value(root_dir + "post/" + post_weight_names[9])

    ###
    for lid in range(4):
        blockid = lid + 1

        block_1__cross_attention__attention_weights__fc__bias_file = root_dir + "block_"+str(blockid)+".cross_attention.attention_weights.fc.bias.1-24"
        block_1__cross_attention__attention_weights__fc__bias = get_value(block_1__cross_attention__attention_weights__fc__bias_file)
        cross_attention__attention_weights__fc__bias_list.append(block_1__cross_attention__attention_weights__fc__bias)
        
        block_1__cross_attention__attention_weights__fc__weight_file = root_dir + "block_"+str(blockid)+".cross_attention.attention_weights.fc.weight.256-24"
        block_1__cross_attention__attention_weights__fc__weight = get_value(block_1__cross_attention__attention_weights__fc__weight_file)
        cross_attention__attention_weights__fc__weight_list.append(block_1__cross_attention__attention_weights__fc__weight)
        
        block_1__cross_attention_norm__ln__bias_file = root_dir + "block_"+str(blockid)+".cross_attention_norm.ln.bias.1-256"
        block_1__cross_attention_norm__ln__bias = get_value(block_1__cross_attention_norm__ln__bias_file)
        cross_attention_norm__ln__bias_list.append(block_1__cross_attention_norm__ln__bias)
        
        block_1__cross_attention_norm__ln__weight_file = root_dir + "block_"+str(blockid)+".cross_attention_norm.ln.weight.1-256"
        block_1__cross_attention_norm__ln__weight = get_value(block_1__cross_attention_norm__ln__weight_file)
        cross_attention_norm__ln__weight_list.append(block_1__cross_attention_norm__ln__weight)

        block_1__cross_attention__output_proj__fc__bias_file = root_dir + "block_"+str(blockid)+".cross_attention.output_proj.fc.bias.1-256"
        block_1__cross_attention__output_proj__fc__bias = get_value(block_1__cross_attention__output_proj__fc__bias_file)
        cross_attention__output_proj__fc__bias_list.append(block_1__cross_attention__output_proj__fc__bias)

        block_1__cross_attention__output_proj__fc__weight_file = root_dir + "block_"+str(blockid)+".cross_attention.output_proj.fc.weight.256-256"
        block_1__cross_attention__output_proj__fc__weight = get_value(block_1__cross_attention__output_proj__fc__weight_file)
        cross_attention__output_proj__fc__weight_list.append(block_1__cross_attention__output_proj__fc__weight)

        block_1__cross_attention__position_encoder__fc1__bias_file = root_dir + "block_"+str(blockid)+".cross_attention.position_encoder.fc1.bias.1-256"
        block_1__cross_attention__position_encoder__fc1__bias = get_value(block_1__cross_attention__position_encoder__fc1__bias_file)
        cross_attention__position_encoder__fc1__bias_list.append(block_1__cross_attention__position_encoder__fc1__bias)

        block_1__cross_attention__position_encoder__fc1__weight_file = root_dir + "block_"+str(blockid)+".cross_attention.position_encoder.fc1.weight.3-256"
        block_1__cross_attention__position_encoder__fc1__weight = get_value(block_1__cross_attention__position_encoder__fc1__weight_file)
        cross_attention__position_encoder__fc1__weight_list.append(block_1__cross_attention__position_encoder__fc1__weight)

        block_1__cross_attention__position_encoder__fc2__bias_file = root_dir + "block_"+str(blockid)+".cross_attention.position_encoder.fc2.bias.1-256"
        block_1__cross_attention__position_encoder__fc2__bias = get_value(block_1__cross_attention__position_encoder__fc2__bias_file)
        cross_attention__position_encoder__fc2__bias_list.append(block_1__cross_attention__position_encoder__fc2__bias)

        block_1__cross_attention__position_encoder__fc2__weight_file = root_dir + "block_"+str(blockid)+".cross_attention.position_encoder.fc2.weight.256-256"
        block_1__cross_attention__position_encoder__fc2__weight = get_value(block_1__cross_attention__position_encoder__fc2__weight_file)
        cross_attention__position_encoder__fc2__weight_list.append(block_1__cross_attention__position_encoder__fc2__weight)

        block_1__cross_attention__position_encoder__ln1__bias_file = root_dir + "block_"+str(blockid)+".cross_attention.position_encoder.ln1.bias.1-256"
        block_1__cross_attention__position_encoder__ln1__bias = get_value(block_1__cross_attention__position_encoder__ln1__bias_file)
        cross_attention__position_encoder__ln1__bias_list.append(block_1__cross_attention__position_encoder__ln1__bias)

        block_1__cross_attention__position_encoder__ln1__weight_file = root_dir + "block_"+str(blockid)+".cross_attention.position_encoder.ln1.weight.1-256"
        block_1__cross_attention__position_encoder__ln1__weight = get_value(block_1__cross_attention__position_encoder__ln1__weight_file)
        cross_attention__position_encoder__ln1__weight_list.append(block_1__cross_attention__position_encoder__ln1__weight)

        block_1__cross_attention__position_encoder__ln2__bias_file = root_dir + "block_"+str(blockid)+".cross_attention.position_encoder.ln2.bias.1-256"
        block_1__cross_attention__position_encoder__ln2__bias = get_value(block_1__cross_attention__position_encoder__ln2__bias_file)
        cross_attention__position_encoder__ln2__bias_list.append(block_1__cross_attention__position_encoder__ln2__bias)

        block_1__cross_attention__position_encoder__ln2__weight_file = root_dir + "block_"+str(blockid)+".cross_attention.position_encoder.ln2.weight.1-256"
        block_1__cross_attention__position_encoder__ln2__weight = get_value(block_1__cross_attention__position_encoder__ln2__weight_file)
        cross_attention__position_encoder__ln2__weight_list.append(block_1__cross_attention__position_encoder__ln2__weight)

        block_1__ffn__fc1__bias_file = root_dir + "block_"+str(blockid)+".ffn.fc1.bias.1-512"
        block_1__ffn__fc1__bias = get_value(block_1__ffn__fc1__bias_file)
        ffn__fc1__bias_list.append(block_1__ffn__fc1__bias)

        block_1__ffn__fc1__weight_file = root_dir + "block_"+str(blockid)+".ffn.fc1.weight.256-512"
        block_1__ffn__fc1__weight = get_value(block_1__ffn__fc1__weight_file)
        ffn__fc1__weight_list.append(block_1__ffn__fc1__weight)

        block_1__ffn__fc2__bias_file = root_dir + "block_"+str(blockid)+".ffn.fc2.bias.1-256"
        block_1__ffn__fc2__bias = get_value(block_1__ffn__fc2__bias_file)
        ffn__fc2__bias_list.append(block_1__ffn__fc2__bias)

        block_1__ffn__fc2__weight_file = root_dir + "block_"+str(blockid)+".ffn.fc2.weight.512-256"
        block_1__ffn__fc2__weight = get_value(block_1__ffn__fc2__weight_file)
        ffn__fc2__weight_list.append(block_1__ffn__fc2__weight)

        block_1__ffn_norm__ln__bias_file = root_dir + "block_"+str(blockid)+".ffn_norm.ln.bias.1-256"
        block_1__ffn_norm__ln__bias = get_value(block_1__ffn_norm__ln__bias_file)
        ffn_norm__ln__bias_list.append(block_1__ffn_norm__ln__bias)

        block_1__ffn_norm__ln__weight_file = root_dir + "block_"+str(blockid)+".ffn_norm.ln.weight.1-256"
        block_1__ffn_norm__ln__weight = get_value(block_1__ffn_norm__ln__weight_file)
        ffn_norm__ln__weight_list.append(block_1__ffn_norm__ln__weight)

        block_1__mh_attention__key__bias_file = root_dir + "block_"+str(blockid)+".mh_attention.key.bias.1-256"
        block_1__mh_attention__key__bias = get_value(block_1__mh_attention__key__bias_file)
        mh_attention__key__bias_list.append(block_1__mh_attention__key__bias)

        block_1__mh_attention__key__weight_file = root_dir + "block_"+str(blockid)+".mh_attention.key.weight.256-256"
        block_1__mh_attention__key__weight = get_value(block_1__mh_attention__key__weight_file)
        mh_attention__key__weight_list.append(block_1__mh_attention__key__weight)

        block_1__mh_attention__query__bias_file = root_dir + "block_"+str(blockid)+".mh_attention.query.bias.1-256"
        block_1__mh_attention__query__bias = get_value(block_1__mh_attention__query__bias_file)
        mh_attention__query__bias_list.append(block_1__mh_attention__query__bias)

        block_1__mh_attention__query__weight_file = root_dir + "block_"+str(blockid)+".mh_attention.query.weight.256-256"
        block_1__mh_attention__query__weight = get_value(block_1__mh_attention__query__weight_file)
        mh_attention__query__weight_list.append(block_1__mh_attention__query__weight)

        block_1__mh_attention__value__bias_file = root_dir + "block_"+str(blockid)+".mh_attention.value.bias.1-256"
        block_1__mh_attention__value__bias = get_value(block_1__mh_attention__value__bias_file)
        mh_attention__value__bias_list.append(block_1__mh_attention__value__bias)

        block_1__mh_attention__value__weight_file = root_dir + "block_"+str(blockid)+".mh_attention.value.weight.256-256"
        block_1__mh_attention__value__weight = get_value(block_1__mh_attention__value__weight_file)
        mh_attention__value__weight_list.append(block_1__mh_attention__value__weight)

        block_1__mh_attention__out__bias_file = root_dir + "block_"+str(blockid)+".mh_attention.out.bias.1-256"
        block_1__mh_attention__out__bias = get_value(block_1__mh_attention__out__bias_file)
        mh_attention__out__bias_list.append(block_1__mh_attention__out__bias)

        block_1__mh_attention__out__weight_file = root_dir + "block_"+str(blockid)+".mh_attention.out.weight.256-256"
        block_1__mh_attention__out__weight = get_value(block_1__mh_attention__out__weight_file)
        mh_attention__out__weight_list.append(block_1__mh_attention__out__weight)

        block_1__mh_attention_norm__ln__bias_file = root_dir + "block_"+str(blockid)+".mh_attention_norm.ln.bias.1-256"
        block_1__mh_attention_norm__ln__bias = get_value(block_1__mh_attention_norm__ln__bias_file)
        mh_attention_norm__ln__bias_list.append(block_1__mh_attention_norm__ln__bias)

        block_1__mh_attention_norm__ln__weight_file = root_dir + "block_"+str(blockid)+".mh_attention_norm.ln.weight.1-256"
        block_1__mh_attention_norm__ln__weight = get_value(block_1__mh_attention_norm__ln__weight_file)
        mh_attention_norm__ln__weight_list.append(block_1__mh_attention_norm__ln__weight)

        block_1__reg_branches__fc1__bias_file = root_dir + "block_"+str(blockid)+".reg_branches.fc1.bias.1-256"
        block_1__reg_branches__fc1__bias = get_value(block_1__reg_branches__fc1__bias_file)
        reg_branches__fc1__bias_list.append(block_1__reg_branches__fc1__bias)

        block_1__reg_branches__fc1__weight_file = root_dir + "block_"+str(blockid)+".reg_branches.fc1.weight.256-256"
        block_1__reg_branches__fc1__weight = get_value(block_1__reg_branches__fc1__weight_file)
        reg_branches__fc1__weight_list.append(block_1__reg_branches__fc1__weight)

        block_1__reg_branches__fc2__bias_file = root_dir + "block_"+str(blockid)+".reg_branches.fc2.bias.1-256"
        block_1__reg_branches__fc2__bias = get_value(block_1__reg_branches__fc2__bias_file)
        reg_branches__fc2__bias_list.append(block_1__reg_branches__fc2__bias)

        block_1__reg_branches__fc2__weight_file = root_dir + "block_"+str(blockid)+".reg_branches.fc2.weight.256-256"
        block_1__reg_branches__fc2__weight = get_value(block_1__reg_branches__fc2__weight_file)
        reg_branches__fc2__weight_list.append(block_1__reg_branches__fc2__weight)

        block_1__reg_branches__fc3__bias_file = root_dir + "block_"+str(blockid)+".reg_branches.fc3.bias.1-8"
        block_1__reg_branches__fc3__bias = get_value(block_1__reg_branches__fc3__bias_file)
        reg_branches__fc3__bias_list.append(block_1__reg_branches__fc3__bias)

        block_1__reg_branches__fc3__weight_file = root_dir + "block_"+str(blockid)+".reg_branches.fc3.weight.256-8"
        block_1__reg_branches__fc3__weight = get_value(block_1__reg_branches__fc3__weight_file)
        reg_branches__fc3__weight_list.append(block_1__reg_branches__fc3__weight)
    
    gen_sv_transformer_decoder()


#Owed by: http://blog.csdn.net/chunyexiyu
#direct get the input name from called function code
import inspect
def retrieve_name_ex(var):
    stacks = inspect.stack()
    try:
        callFunc = stacks[1].function
        code = stacks[2].code_context[0]
        startIndex = code.index(callFunc)
        startIndex = code.index("(", startIndex + len(callFunc)) + 1
        endIndex = code.index(")", startIndex)
        return code[startIndex:endIndex].strip()
    except:
        return ""

def outputVar(var, flag='_t'):
    return "{}{}".format(retrieve_name_ex(var), flag)

if __name__ == '__main__':
    cross_attention__attention_weights__fc__bias_list=[]
    cross_attention__attention_weights__fc__weight_list=[]
    cross_attention_norm__ln__bias_list=[]
    cross_attention_norm__ln__weight_list=[]
    cross_attention__output_proj__fc__bias_list=[]
    cross_attention__output_proj__fc__weight_list=[]
    cross_attention__position_encoder__fc1__bias_list=[]
    cross_attention__position_encoder__fc1__weight_list=[]
    cross_attention__position_encoder__fc2__bias_list=[]
    cross_attention__position_encoder__fc2__weight_list=[]
    cross_attention__position_encoder__ln1__bias_list=[]
    cross_attention__position_encoder__ln1__weight_list=[]
    cross_attention__position_encoder__ln2__bias_list=[]
    cross_attention__position_encoder__ln2__weight_list=[]
    ffn__fc1__bias_list=[]
    ffn__fc1__weight_list=[]
    ffn__fc2__bias_list=[]
    ffn__fc2__weight_list=[]
    ffn_norm__ln__bias_list=[]
    ffn_norm__ln__weight_list=[]
    mh_attention__key__bias_list=[]
    mh_attention__key__weight_list=[]
    mh_attention__query__bias_list=[]
    mh_attention__query__weight_list=[]
    mh_attention__value__bias_list=[]
    mh_attention__value__weight_list=[]
    mh_attention__out__bias_list=[]
    mh_attention__out__weight_list=[]
    mh_attention_norm__ln__bias_list=[]
    mh_attention_norm__ln__weight_list=[]
    reg_branches__fc1__bias_list=[]
    reg_branches__fc1__weight_list=[]
    reg_branches__fc2__bias_list=[]
    reg_branches__fc2__weight_list=[]
    reg_branches__fc3__bias_list=[]
    reg_branches__fc3__weight_list=[]

    main()  
 
   
    # aa = dict(
    #   reg_branches__fc1__bias_list=1,
    #   reg_branches__fc1__weight_list=2,
    #   reg_branches__fc2__bias_list=3,
    #   reg_branches__fc2__weight_list=33,
    #   reg_branches__fc3__bias_list=4,
    #   reg_branches__fc3__weight_list=5
    # )
    # # print(aa.keys())
    # for key, val in aa.items():
    #   locals()[key+'_t'] = val 
    

    # print(reg_branches__fc3__weight_list_t)
