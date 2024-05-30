# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2022-03-20 11:09:13
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2022-03-20 11:09:13
#  **************************************************************/

from hashlib import sha1
import onnx_graphsurgeon as gs
import onnx
from loguru import logger as LOG
from utils import model_check_and_save
import os
import numpy as np

def get_node(onnx_path, node_name_list):
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    graph.cleanup(remove_unused_node_outputs =True, remove_unused_graph_inputs=True)

    tmap = graph.tensors()
    node_list_1 = []
    for i in range(len(node_name_list)):
        for key, value in tmap.items():
            if key == node_name_list[i]:
                node_list_1.append(value)
    return model, graph, tmap, node_list_1

def mergemodel(onnx_path_1,
               model1_node_output_name_list,
               onnx_path_2,
               model2_node_input_name_list,
               model_save_path):
    _, graph1, _,  model_node_list_1 = get_node(
        onnx_path=onnx_path_1, node_name_list=model1_node_output_name_list)
    _, graph2, _, model_node_list_2 = get_node(
        onnx_path=onnx_path_2, node_name_list=model2_node_input_name_list)

    print('>>model_node_list_1[0] : ', model_node_list_1[0])
    print('>>>model_node_list_2[0] : ', model_node_list_2[0])
    model_check_and_save(graph1, "svdet0727_vru512_folded_SS.onnx")

    # sh = gs.Constant(name="img_shape", values=np.array([[192,736]]))
    # graph2.inputs[4] = sh

    input = [graph1.inputs[0], graph2.inputs[3], graph2.inputs[4]]
    output = graph2.outputs
    print('>>>>model_node_list_1[0] : ', input)
    print('>>>>model_node_list_1[0] : ', output)
  
    node_all = []
    for node1 in graph1.nodes:
        node_all.append(node1)

    for node2 in graph2.nodes:
        node_all.append(node2)

    for idx in range(3):
      for it in node_all:
          for outp in it.outputs:
              if outp.name == model1_node_output_name_list[idx]:
                  node_tmp = it.outputs[0]
                  print('\n\n ------ node_tmp ---------\n ', node_tmp)
    
      for it in node_all:
          for i in range(len(it.inputs)):
              if it.inputs[i].name == model2_node_input_name_list[idx]:
                  it.inputs[i] = node_tmp
                  print("it.inputs : \n", it.inputs)

    graph = gs.Graph(nodes=node_all, inputs=input, outputs=output)
    model_check_and_save(graph, model_save_path)


def main():
    BASE_PATH = '/home/igs/transformer/FasterTransformer-main/0727_src_model/'

    onnx_path_0     = BASE_PATH + "svdet0727_vru512_folded_S.onnx"
    
    onnx_path_1     = BASE_PATH + "sv_tf_decoder_e2e0907.onnx"
    last_onnx   = BASE_PATH + "merge.onnx"    

    mergemodel(onnx_path_1=onnx_path_0,
               model1_node_output_name_list=['11503', '11518', '11521'],
               onnx_path_2=onnx_path_1,
               model2_node_input_name_list=['value_0', 'value_1', 'value_2'],
               model_save_path=last_onnx)

if __name__ == '__main__':
  import argparse

  try:    
    parser = argparse.ArgumentParser(description="need one para")
    parser.add_argument("-o", "--out", type=str, required=True, help="last onnx file name")
    args = parser.parse_args()       
  except Exception as e:
    print(e)
  
  main()
