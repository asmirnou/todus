import argparse
import os
import uff
import graphsurgeon as gs
import importlib
import tensorflow as tf
from pathlib import Path
from model import model_path
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2


def convert():
    args = parse_commandline_arguments()

    config = importlib.import_module('config.{}'.format(args.model_name))
    model = config.Model

    a_model_path = model_path(args.model_name, pretrained=args.pretrained)
    pipeline_path = os.path.join(a_model_path, 'pipeline.config')
    graph_path = os.path.join(a_model_path, 'frozen_inference_graph.pb')
    print('Converting to UFF:\n\t{}\n\t{}'.format(pipeline_path, graph_path))

    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline, allow_unknown_field=True)

    graph = gs.DynamicGraph(graph_path)
    dynamic_graph = model.unsupported_nodes_to_plugin_nodes(graph,
                                                            pipeline.model.ssd.num_classes + 1)

    output_filename = os.path.abspath(os.path.join(Path(graph_path).parent.parent,
                                                   args.model_name + '.uff'))
    uff.from_tensorflow(dynamic_graph.as_graph_def(),
                        output_nodes=model.OUTPUT_NODES,
                        output_filename=output_filename,
                        text=args.text)


def parse_commandline_arguments():
    parser = argparse.ArgumentParser(description='Converts certain trained TensorFlow models to UFF formaf.')

    parser.add_argument("-m", "--model", dest="model_name", required=True,
                        help="Name of object detection model to convert")
    parser.add_argument("-i", "--inference-graph", dest="pretrained",
                        action="store_false", default=True,
                        help="Use trained inference graph instead of pre-trained")
    parser.add_argument("-t", "--text", dest="text",
                        action="store_true", default=False,
                        help="If set, the converter will also write out a human readable UFF file")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    convert()
