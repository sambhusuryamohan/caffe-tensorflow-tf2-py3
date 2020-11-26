import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib 
def optimize_graph(frozen_model_pb, input_node_name, output_node_name):
    inputGraph = tf.GraphDef()
    with tf.gfile.Open(frozen_model_pb, "rb") as f:
        data2read = f.read()
        inputGraph.ParseFromString(data2read)
      
        outputGraph = optimize_for_inference_lib.optimize_for_inference(
                    inputGraph, [input_node_name], # an array of the input node(s)
                    [output_node_name], # an array of output nodes
                    tf.float32.as_datatype_enum)
                    # Save the optimized graph'test.pb'
        f = tf.gfile.FastGFile('OptimizedGraph.pb', "w")
        f.write(outputGraph.SerializeToString()) 
