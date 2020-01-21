import graphsurgeon as gs
import tensorflow as tf

Input = gs.create_node("Input",
    op="Placeholder",
    dtype=tf.float32,
    shape=[1, 3, 300, 300])
PriorBox = gs.create_plugin_node(name="MultipleGridAnchorGenerator", op="GridAnchor_TRT",
    numLayers=6,
    minSize=0.2,
    maxSize=0.95,
    aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
    variance=[0.1,0.1,0.2,0.2],
    featureMapShapes=[19, 10, 5, 3, 2, 1])
NMS = gs.create_plugin_node(name="NMS", op="NMS_OPT_TRT",
    shareLocation=1,
    varianceEncodedInTarget=0,
    backgroundLabelId=0,
    confidenceThreshold=0.3,
    nmsThreshold=0.6,
    topK=100,
    keepTopK=100,
    numClasses=91,
    inputOrder=[0, 7, 6],
    confSigmoid=1,
    confSoftmax=0,
    isNormalized=1,
    numLayers=6)
concat_priorbox = gs.create_node(name="concat_priorbox", op="ConcatV2", dtype=tf.float32, axis=2)
#concat_box_loc = gs.create_plugin_node("concat_box_loc", op="FlattenConcat_TRT", dtype=tf.float32, axis=1, ignoreBatch=0)
#concat_box_conf = gs.create_plugin_node("concat_box_conf", op="FlattenConcat_TRT", dtype=tf.float32, axis=1, ignoreBatch=0)

namespace_plugin_map = {
    "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
    "MultipleGridAnchorGenerator": PriorBox,
    "Postprocessor": NMS,
    "image_tensor": Input,
    "ToFloat": Input,
    "Preprocessor": Input,
#    "concat": concat_box_loc,
#    "concat_1": concat_box_conf
}

def preprocess(dynamic_graph):
    dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_op("Identity"))
    dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("Squeeze"))
    dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("concat"))
    dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("concat_1"))

    for i in range(0,6):
        dynamic_graph.remove(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/stack".format(i)))
        dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/Reshape".format(i)))
        dynamic_graph.remove(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/stack_1".format(i)))
        dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/Reshape_1".format(i)))
        dynamic_graph.remove(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/Shape".format(i)))

    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (Postprocessor).
    dynamic_graph.remove(dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)
    # Disconnect the Input node from NMS.
    dynamic_graph.find_nodes_by_op("NMS_OPT_TRT")[0].input.remove("Input")
    # Disconnect concat/axis and concat_1/axis from NMS.
    dynamic_graph.find_nodes_by_op("NMS_OPT_TRT")[0].input.remove("concat/axis")
    dynamic_graph.find_nodes_by_op("NMS_OPT_TRT")[0].input.remove("concat_1/axis")
    dynamic_graph.find_nodes_by_name("Input")[0].input.remove("image_tensor:0")

