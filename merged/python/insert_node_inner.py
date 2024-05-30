# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2022-09-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2022-09-03 11:09:48
#  **************************************************************/

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from loguru import logger

def make_node(type, name, input, output, **kwargs):
    new_node = onnx.helper.make_node(
        op_type=type,
        name=name,
        #domain='ai.onnx.contrib',
        inputs=input,  # ["842"],
        outputs=output,  # ['rm_output_new']
        #value=onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [4], [1, 1, 1.81, 1.81])
        **kwargs
    )
    return new_node

def find_node(node, optype, name):
    for i in range(len(node)):
        if node[i].op_type == optype and node[i].name == name:
            node_rise = node[i]
            return i, node_rise

def del_nodes(node, idx):
    total_num = len(node)
    for i in range(total_num):
        j = total_num - 1 - i  
        if j > idx:
            continue

        node.remove(node[j])

def del_node(node, idx):
    node.remove(node[idx])

def update_input(graph):
    total_out_num = len(graph.input)
    for i in range(total_out_num):
        j = total_out_num - 1 - i
        graph.input.remove(graph.input[j])

def update_out(graph):
    total_out_num = len(graph.output)
    for i in range(total_out_num):
        j = total_out_num - 1 - i
        graph.output.remove(graph.output[j])

    Y = helper.make_tensor_value_info(
        'l_boxes', TensorProto.FLOAT, [1, 512,8])
    Z = helper.make_tensor_value_info(
        'l_scores', TensorProto.FLOAT, [1, 512, 5])
    graph.output.append(Y)
    graph.output.append(Z)

def modify_out(bpp, onnx_file, onnx_new_file):
    onnx_model = onnx.load(bpp + onnx_file)
    graph = onnx_model.graph
    node = graph.node

    new1 = make_node("Identity", name="Identity_1", input=["reg_out"], output=["l_boxes"])
    new2 = make_node("Identity", name="Identity_2", input=["cls_out"], output=["l_scores"])

    update_out(graph)

    graph.node.append(new1)
    graph.node.append(new2)
    logger.info("node :{}, {}".format(len(node), node[len(node)-1]))
    onnx.save(onnx_model, bpp + onnx_new_file)

def main(bpp, onnx_file, onnx_new_file):
    modify_out(bpp, onnx_file, onnx_new_file)

if __name__ == "__main__":
    logger.info('onnx version:{}'.format(onnx.__version__))
    BASE_PATH = '/home/igs/transformer/FasterTransformer-main/0727_src_model/'
    ONNX = "merge.onnx"
    NEW_NAME = "merge_identify.onnx"

    main(bpp = BASE_PATH, onnx_file = ONNX, onnx_new_file = NEW_NAME)
    logger.info("done")
