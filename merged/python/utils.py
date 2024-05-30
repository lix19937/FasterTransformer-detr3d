import onnx_graphsurgeon as gs
import onnx
import netron 
import argparse
from loguru import logger as LOG

def model_check_and_save(graph, save_path, check = False):
  model = gs.export_onnx(graph)
  if check:
    print('check:', onnx.checker.check_model(model))

  # model.ir_version = 6
  # onnx.helper.make_model(graph_def, ir_version = 7)  
  # LOG.info('ir:{}'.format(model.ir_version))

  onnx.save(model, save_path)
  LOG.info('ir:{}'.format(model.ir_version))

  print('save done ')
  # netron.start(save_path)

def print_attribute_byonnx(model):
  print("----------- def in  onnx.in.proto -----------\n")
  print("================= model ==================\n")
  print("model.ir_version  :    \n", model.ir_version)          # 7
  print("model.opset_import:    \n", model.opset_import)  
  print("model.producer_name:   \n", model.producer_name)       # 无
  print("model.producer_version:\n", model.producer_version)    # 无
  print("model.domain:          \n", model.domain)              # 无
  print("model.model_version:   \n", model.model_version)       # 0
  print("model.doc_string:      \n", model.doc_string)          # 无
  print("model.graph:           \n", model.graph)               #在graph中打印
  print("model.metadata_props:  \n", model.metadata_props)      # 空

  print("================= graph ==================\n")
  # graph = gs.import_onnx(model)                    #使用的是onnx gs方法导入
  graph = model.graph                                #使用的是onnx 方法导入，二者导入后graph,node的包含的操作是不同的
  print("graph.name:            \n", graph.name)                         # onnx_graphsurgeon_graph
  print("graph.node:            \n", graph.node)
  # print("graph.initializer:     \n", graph.initializer)         #一堆乱码
  print("graph.doc_string:      \n", graph.doc_string)             #空
  print("graph.input:           \n", graph.input)
  print("graph.input[0].name:   \n", graph.input[0].name)
  print("graph.input[0].type:   \n", graph.input[0].type)
  print("graph.output:          \n", graph.output)
  print("graph.value_info:      \n", graph.value_info)

  print("graph.input[0].type.tensor_type:           \n", graph.input[0].type.tensor_type)
  print("graph.input[0].type.tensor_type.elem_type: \n", graph.input[0].type.tensor_type.elem_type)
  print("graph.input[0].type.tensor_type.shape:     \n", graph.input[0].type.tensor_type.shape)
  print("graph.input[0].type.tensor_type.shape.dim: \n", graph.input[0].type.tensor_type.shape.dim)
  print("graph.input[0].type.tensor_type.shape.dim[0].dim_value: \n", graph.input[0].type.tensor_type.shape.dim[0].dim_value)
 

  print("================= node ==================\n")
  print("graph.node[0].name:     \n", graph.node[0].name)
  print("graph.node[0].input:    \n", graph.node[0].input)
  print("graph.node[0].input[0]: \n", graph.node[0].input[0])
  print("graph.node[0].output:   \n", graph.node[0].output)
  print("graph.node[0].op_type:  \n", graph.node[0].op_type)
  print("graph.node[0].domain:   \n", graph.node[0].domain)
  print("graph.node[0].attribute:\n", graph.node[0].attribute)
  print("graph.node[0].attribute[0].name: \n", graph.node[0].attribute[0].name)
  print("graph.node[0].attribute[0].i:    \n", graph.node[0].attribute[0].i)
  print("graph.node[0].attribute[0].type, \n", graph.node[0].attribute[0].type)

def print_attribute_byonnxgs(model):
  # print("================= model ==================\n")
  # print("model.ir_version  :    \n", model.ir_version)          # 7
  # print("model.opset_import:    \n", model.opset_import)  
  # print("model.producer_name:   \n", model.producer_name)       # 无
  # print("model.producer_version:\n", model.producer_version)    # 无
  # print("model.domain:          \n", model.domain)              # 无
  # print("model.model_version:   \n", model.model_version)       # 0
  # print("model.doc_string:      \n", model.doc_string)          # 无
  # print("model.graph:           \n", model.graph)               #在graph中打印
  # print("model.metadata_props:  \n", model.metadata_props)      # 空

  print("================= graph ==================\n")
  graph = gs.import_onnx(model)                    #使用的是onnx gs方法导入
  print("graph.name:              \n", graph.name)                         # onnx_graphsurgeon_graph
  print("graph.nodes:             \n", graph.nodes)
  print("graph.inputs:            \n", graph.inputs)
  print("graph.outputs:           \n", graph.outputs)
  print("graph.doc_string:        \n", graph.doc_string)   

  print("graph.inputs[0].name:    \n", graph.inputs[0].name)
  print("graph.inputs[0].dtype:   \n", graph.inputs[0].dtype)
  print("graph.inputs[0].shape:   \n", graph.inputs[0].shape)

  print("================= node ==================\n")
  print("graph.nodes[0].name:            \n", graph.nodes[0].name)
  print("graph.nodes[0].inputs:          \n", graph.nodes[0].inputs)
  print("graph.nodes[0].outputs:         \n", graph.nodes[0].outputs)
  print("graph.nodes[0].op:              \n", graph.nodes[0].op)
  print("graph.nodes[0].attrs:              \n", graph.nodes[0].attrs)
  print("graph.nodes[0].attrs['transA']: \n", graph.nodes[0].attrs['transA'])
  print("graph.nodes[0].attrs['transB']: \n", graph.nodes[0].attrs['transB'])
 
  print("graph.nodes[0].inputs[0]:       \n", graph.nodes[0].inputs[0])
  print("graph.nodes[0].inputs[0].name:  \n", graph.nodes[0].inputs[0].name)
  print("graph.nodes[0].inputs[0].shape: \n", graph.nodes[0].inputs[0].shape)
  print("graph.nodes[0].inputs[0].dtype: \n", graph.nodes[0].inputs[0].dtype)
   
  print("graph.nodes[0].inputs[1]:       \n", graph.nodes[0].inputs[1])
  print("graph.nodes[0].inputs[1].name:  \n", graph.nodes[0].inputs[1].name)
  print("graph.nodes[0].inputs[1].shape: \n", graph.nodes[0].inputs[1].shape)
  print("graph.nodes[0].inputs[1].dtype: \n", graph.nodes[0].inputs[1].dtype)

def get_tensor_name_dtype_shape(tmap, tensors_name_list):
  name_list = []
  datatype_list = []
  shape_list = []
  for key, value in tmap.items():
    for tensor in tensors_name_list:
      if(key == tensor):
        name_list.append(value.name)
        datatype_list.append(value.dtype)
        print("value.shape : \n", value.shape)
        if value.shape is None:
          value.shape = [gs.Tensor.DYNAMIC]
        # else:
        #   ## set value.shape[i] is int
        #   for i in range(len(value.shape)):
            # if isinstance(value.shape[i], str):
            #   print("not int : \n", value.shape[i])
            #   value.shape[i] = gs.Tensor.DYNAMIC

        shape_list.append(value.shape)
  return name_list, datatype_list, shape_list

################# set command line parameters ##############
def parse_args():
  parser = argparse.ArgumentParser('you should add those parameter')
  parser.add_argument('--onnx_path', default=' ', help='The path of onnx model')
  parser.add_argument('--input_node_name', default=' ', help='The start position of the new node')
  parser.add_argument('--output_node_name', default=' ', help='The end position  of the new node')
  parser.add_argument('--op', default=' ', help=' The ONNX op to use for the new node')
  parser.add_argument('--name', default=' ', help=' The name to use for the new node')
  parser.add_argument('--attrs', default=' ', help='Attributes to set in the new node. Format: --attrs <name>=value.')
  parser.add_argument('--inputs_variant', default=' ', help='Additional input: --inputs_variant <name>=value.')
  parser.add_argument('-o','--save_path', default=' ', help='Path to save the ONNX model')

  args = parser.parse_args()
  return args

