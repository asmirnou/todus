import argparse
import os
import uff
import graphsurgeon as gs
import importlib
from pathlib import Path
from model import model_path


def convert():
    args = parse_commandline_arguments()

    config = importlib.import_module('config.{}'.format(args.model_name))
    model = config.Model

    graph_path = os.path.join(model_path(args.model_name), 'frozen_inference_graph.pb')
    graph = gs.DynamicGraph(graph_path)
    dynamic_graph = model.unsupported_nodes_to_plugin_nodes(graph)

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
    parser.add_argument("-t", "--text", dest="text",
                        action="store_true", default=False,
                        help="If set, the converter will also write out a human readable UFF file")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    convert()
