import os
import json
import sys
import math
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from tempfile import NamedTemporaryFile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from model import where_is_dataset, model_path


class TensorFlowInference(object):
    def __init__(self, pb_model_path):
        self.__detection_graph = tf.Graph()
        self.__sess = tf.compat.v1.Session(graph=self.__detection_graph)

        with self.__detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(pb_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def infer(self, image_path):
        image = Image.open(image_path).convert('RGB')
        (im_width, im_height) = image.size
        boxes, label_codes, scores = self._run_tensorflow_graph(np.array(image))

        boxes[:, 0] = np.multiply(boxes[:, 0], im_height)  # y_min
        boxes[:, 1] = np.multiply(boxes[:, 1], im_width)  # x_min
        boxes[:, 2] = np.multiply(boxes[:, 2], im_height)  # y_max
        boxes[:, 3] = np.multiply(boxes[:, 3], im_width)  # x_max

        boxes[:, 2] = np.subtract(boxes[:, 2], boxes[:, 0])  # height
        boxes[:, 3] = np.subtract(boxes[:, 3], boxes[:, 1])  # width

        return boxes, label_codes, scores

    def _run_tensorflow_graph(self, image_np):
        ops = self.__detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.__detection_graph.get_tensor_by_name(tensor_name)

        image_tensor = self.__detection_graph.get_tensor_by_name('image_tensor:0')
        output_dict = self.__sess.run(tensor_dict,
                                      feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

        boxes = output_dict['detection_boxes'][0]
        label_codes = output_dict['detection_classes'][0].astype(np.uint8)
        scores = output_dict['detection_scores'][0]

        return boxes, label_codes, scores


def parse_commandline_arguments():
    parser = argparse.ArgumentParser(description='Converts certain trained TensorFlow models to UFF formaf.')

    parser.add_argument("-m", "--model", dest="model_name", required=True,
                        help="Name of object detection model to convert")
    parser.add_argument("-c", "--categories", nargs='+', dest="categories", default=[],
                        help="List of category names separated by spaces, e.g. -c person dog bicycle")
    parser.add_argument("-i", "--inference-graph", dest="pretrained",
                        action="store_false", default=True,
                        help="Use trained inference graph instead of pre-trained")

    args = parser.parse_args()
    return args


def print_progress(pct_done):
    isatty = sys.stdout.isatty()
    clear_char = "\r" if isatty else ""
    endl_char = "" if isatty else "\n"
    progress_bar_width = int(math.floor(pct_done * 50 / 100.0))
    sys.stdout.write("{}Evaluation progress [{}{}] {:.1f}%{}".format(
        clear_char,
        "=" * progress_bar_width,
        " " * (50 - progress_bar_width),
        pct_done,
        endl_char))
    sys.stdout.flush()


def main():
    args = parse_commandline_arguments()

    dataType = 'val2017'
    dataset_path = where_is_dataset()
    annotation_file = os.path.join(dataset_path,
                                   'raw-data/annotations/instances_{}.json'.format(dataType))
    coco_gt = COCO(annotation_file)

    a_model_path = model_path(args.model_name, pretrained=args.pretrained)
    graph_path = os.path.join(a_model_path, 'frozen_inference_graph.pb')
    print('Evaluating:\n\t{}'.format(graph_path))

    inference = TensorFlowInference(graph_path)

    img_ids = set()
    cat_ids = coco_gt.getCatIds(catNms=args.categories)
    for cat_id in cat_ids:
        img_ids.update(coco_gt.getImgIds(catIds=[cat_id]))
    img_ids = list(img_ids)

    imgs = coco_gt.loadImgs(img_ids)

    results = list()
    for progress, img in enumerate(imgs):
        img_id = img['id']
        img_filename = img['file_name']

        image_path = os.path.join(dataset_path, 'raw-data', dataType, img_filename)
        try:
            boxes, label_codes, scores = inference.infer(image_path)
        except Exception as e:
            print(image_path)
            raise e

        for idx, score in enumerate(scores):
            if score <= 0.:
                continue

            result = dict()
            result['image_id'] = img_id
            result['category_id'] = int(label_codes[idx])
            result['bbox'] = [round(float(boxes[idx][1]), 2),
                              round(float(boxes[idx][0]), 2),
                              round(float(boxes[idx][3]), 2),
                              round(float(boxes[idx][2]), 2)]
            result['score'] = round(float(score), 2)
            results.append(result)

        print_progress(100 * progress / len(imgs))
    print_progress(100)

    with NamedTemporaryFile(mode='w+', suffix='.json') as res_file:
        json.dump(results, res_file)
        res_file.flush()

        coco_dt = coco_gt.loadRes(res_file.name)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
