import os
import shutil
import tarfile
import requests
import argparse
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils.label_map_util import load_labelmap


def where_is_model():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')


def where_is_dataset():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dataset', 'mscoco')


def model_path(model_name, model_dir=where_is_model(), pretrained=True):
    return os.path.join(model_dir, model_name, "pretrained" if pretrained else "inference-graph")


def download_model(model_name,
                   website='http://download.tensorflow.org/models/object_detection/{}.tar.gz'):
    """Downloads model_name from Tensorflow model zoo.

    Args:
        model_name (str): chosen object detection model
        website (str): website to download model from
    """
    print("Preparing pretrained model {}".format(model_name))

    model_dir = where_is_model()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_url = website.format(model_name)
    model_archive_path = os.path.join(model_dir, "{}.tar.gz".format(model_name))
    model_extract_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_archive_path):
        print("Downloading {}".format(model_url))
        with open(model_archive_path, "wb") as f:
            response = requests.get(model_url, stream=True)
            f.write(response.content)
        print("Download complete")

    print("Unpacking {}".format(model_archive_path))
    with tarfile.open(model_archive_path, "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=model_extract_path)

    a_model_path = model_path(model_name, model_dir)
    shutil.rmtree(a_model_path, ignore_errors=True)
    os.replace(os.path.join(model_extract_path, model_name), a_model_path)
    print("Extracting complete")


def prepare_model(model_name, dataset_type, quantization=False):
    a_model_path = model_path(model_name)
    config_path = os.path.join(a_model_path, 'pipeline.config')
    dataset_path = os.path.join(where_is_dataset(), dataset_type)

    label_map_file = os.path.join(dataset_path, 'mscoco_label_map.pbtxt')
    if not os.path.isfile(label_map_file):
        print("Dataset {} is not ready, prepare dataset at first".format(dataset_type))
        return
    print("Modifying pipeline config {}".format(config_path))
    label_map = load_labelmap(label_map_file)

    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline, allow_unknown_field=True)

    pipeline.train_config.fine_tune_checkpoint = os.path.join(a_model_path, 'model.ckpt')
    pipeline.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline.train_config.from_detection_checkpoint = True
    pipeline.train_input_reader.tf_record_input_reader.input_path[:] = [
        os.path.join(dataset_path, 'coco_train.record-?????-of-00100')]
    pipeline.train_input_reader.label_map_path = \
        os.path.join(dataset_path, 'mscoco_label_map.pbtxt')
    pipeline.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        os.path.join(dataset_path, 'coco_val.record-?????-of-00050')]
    pipeline.eval_input_reader[0].label_map_path = \
        os.path.join(dataset_path, 'mscoco_label_map.pbtxt')
    pipeline.model.ssd.num_classes = max(label_map.item, key=lambda item: item.id).id
    if quantization:  # Quantization Aware Training
        pipeline.graph_rewriter.quantization.delay = 48000
        pipeline.graph_rewriter.quantization.weight_bits = 8
        pipeline.graph_rewriter.quantization.activation_bits = 8

    config_text = text_format.MessageToString(pipeline)
    with tf.io.gfile.GFile(config_path, "wb") as f:
        f.write(config_text)

    print("Model ready")


def parse_commandline_arguments():
    parser = argparse.ArgumentParser(description='Prepares object detection model for training.')

    parser.add_argument("-m", "--model", dest="model_name", required=True,
                        help="Name of object detection model to download and prepare")
    parser.add_argument('-d', "--dataset", dest='dataset_type', required=True,
                        choices=['complete', 'limited'],
                        help='Type of dataset used for training')
    parser.add_argument('-q', "--quantization", dest='quantization',
                        action="store_true", default=False,
                        help='Enable quantization aware training')

    args = parser.parse_args()
    return args


def main():
    args = parse_commandline_arguments()

    download_model(args.model_name)
    prepare_model(args.model_name, args.dataset_type, args.quantization)


if __name__ == '__main__':
    main()
