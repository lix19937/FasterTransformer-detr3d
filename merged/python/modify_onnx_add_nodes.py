# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2022-03-20 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2022-03-20 11:09:48
#  **************************************************************/

from torch.onnx import register_custom_op_symbolic
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from loguru import logger
from onnxsim import simplify

print('onnx version:', onnx.__version__)

BASE_PATH = '/home/igs/transformer/FasterTransformer-main/0727_src_model/'
ONNX     = "svdet0727_vru512_folded.onnx"
NEW_NAME = "svdet0727_vru512_folded_S.onnx"  # reduce one output

def find_node(node, optype, name):
    for i in range(len(node)):
        if node[i].op_type == optype and node[i].name == name:
            node_rise = node[i]
            return i, node_rise


def del_nodes(node, idx):
    total_num = len(node)
    for i in range(total_num):
        j = total_num - 1 - i  # reverse order del
        if j > idx:
            continue
        node.remove(node[j])


def del_nodes_after(node, idx):
    total_num = len(node)
    for i in range(total_num):
        j = total_num - 1 - i  # reverse order del
        if j > idx:            
          node.remove(node[j])


def del_node(node, idx):
    node.remove(node[idx])


def update_input(graph):
    total_out_num = len(graph.input)
    for i in range(total_out_num):
        j = total_out_num - 1 - i
        # print(graph.input[j])
        graph.input.remove(graph.input[j])

def update_input(graph):
    # total_out_num = len(graph.input)
    # for i in range(total_out_num):
    #     j = total_out_num - 1 - i
    #     graph.input.remove(graph.input[j])

    # ------- new create a input  use should
    Y = helper.make_tensor_value_info(
        'zz', TensorProto.INT32, ['b', 4])
    graph.input.append(Y)


def update_out(graph):
    total_out_num = len(graph.output)
    for i in range(total_out_num):
        j = total_out_num - 1 - i
        graph.output.remove(graph.output[j])

    # ------- new create a output  use should
    #Y = helper.make_tensor_value_info('bevout', TensorProto.FLOAT, [1, 64*16, 448, 128])
    # Y = helper.make_tensor_value_info('bevout', TensorProto.FLOAT, [1, 64*16, 256, 256])
    # graph.output.append(Y)


def modify_out():
    onnx_model = onnx.load(BASE_PATH + ONNX)
    graph = onnx_model.graph
    node = graph.node

    idx1, _ = find_node(node, "Conv", "Conv_273")
    logger.info("node num:{}, {}, {}".format(len(node), idx1, node[idx1]))

    idx2, _ = find_node(node, "Conv", "Conv_280")
    logger.info("node num:{}, {}, {}".format(len(node), idx2, node[idx2]))

    idx3, _ = find_node(node, "Conv", "Conv_281")
    logger.info("node num:{}, {}, {}".format(len(node), idx3, node[idx3]))

    # del onnx_model.graph.output[:]
    update_out(graph)
    outputs = [helper.make_tensor_value_info(node[idx1].output[0], TensorProto.FLOAT, shape=(6,256,48,184)),
      helper.make_tensor_value_info(node[idx2].output[0], TensorProto.FLOAT, shape=(6,256,48//2,184//2)),
      helper.make_tensor_value_info(node[idx3].output[0], TensorProto.FLOAT, shape=(6,256,48//4,184//4))]

    for it in outputs:
      graph.output.append(it) 

    logger.info(onnx_model.ir_version)
    onnx_model.ir_version = 6 
    model_simp, check = simplify(onnx_model) 

    onnx.save(model_simp, BASE_PATH + NEW_NAME)


def main():
    modify_out()


if __name__ == "__main__":
    main()
