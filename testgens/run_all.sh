CVTOOLS=/cv2/work/schilkunda/default/cvtools/

INPUT_NODE='Placeholder'
OUTPUT_NODE='split'

./convnet.py

$CVTOOLS/CnnUtils/tensorflow/third_party/freeze_graph.py --input_graph=sample_convnet.pb --input_checkpoint=sample_convnet.ckpt --input_binary=True --output_node_names=$OUTPUT_NODE --output_graph=frozen_convnet.pb

$CVTOOLS/CnnUtils/tensorflow/third_party/graph_transform.py --input_graph=frozen_convnet.pb --output_graph=frozen_convnet_opt.pb --input_node_names=$INPUT_NODE --output_node_names=$OUTPUT_NODE --transforms='fold_constants fold_batch_norms fold_old_batch_norms merge_duplicate_nodes remove_nodes(op=Identity op=CheckNumerics) strip_unused_nodes sort_by_execution_order'

$CVTOOLS/CnnUtils/tensorflow/parse_tf_model.py -p frozen_convnet_opt.pb
