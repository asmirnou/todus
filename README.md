# Todus

A collection of scripts assisting in training TensorFlow 1 [object detection models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) on the [COCO dataset](http://cocodataset.org/).

The models publicly available on the internet have been trained to detect 90 classes of common objects. Sometimes it is worth retraining the models using fewer classes to improve inference speed. The scripts are designed to prepare a data set of only those categories that you need, then assist in training and converting the models to various formats.

## Table of contents
  * [Getting started](#getting-started)
    + [Install dependencies](#install-dependencies)
    + [Directory structure](#directory-structure)
    + [Prepare training dataset](#prepare-training-dataset)
    + [Download and prepare pretrained model](#download-and-prepare-pretrained-model)
    + [Evaluate your model](#evaluate-your-model)
    + [Train the model](#train-the-model)
    + [Export inference graph](#export-inference-graph)
    + [Convert to various formats](#convert-to-various-formats)
  * [Troubleshooting](#troubleshooting)
  * [License](#license)

## Getting started

### Install dependencies

The collection consists of Shell and Python scripts. Install Python requirements in a virtual environment:

```bash
pip install -r requirements.txt
```

Some dependencies are not within the Python Package Index and need to be installed other way.

Refer to the [following](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md) guide to install Object Detection API. Bear in mind that you may need to switch between different versions of the API (see [Troubleshooting](#troubleshooting) section).

`graphsurgeon` and `uff` are optional and require only if you export the models in `.uff` format for TensorRT to run inference on CUDA GPUs. These two packages you can [find](https://developer.nvidia.com/nvidia-tensorrt-7x-download) in TensorRT `.tar` distribution.

```bash
pip install \
    graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl \
    uff/uff-0.6.9-py2.py3-none-any.whl
``` 

### Directory structure

During execution the scripts form the following directory structure:

```
todus
│
└───dataset
│   │
│   └───mscoco
│       │
│       └───complete
│       │   
│       └───limited
│       │
│       └───raw-data
│   
└───model
    │
    └───model-name-1
    │   │
    │   └───inference-graph
    │   │
    │   └───pretrained
    │   │
    │   └───training
    │
    └───model-name-2
        │
        └───...

```

_dataset_ stores a sequence of binary records required for training. The records are prepared either from the _complete_ COCO set of 90 classes or _limited_ to only the classes selected. Downloaded archives of images and ground truth annotations along with their extracted contents will remain in _raw-data_.

_pretrained_ folder of each model keeps downloaded TensorFlow model trained to detect all 90 classes. There you can find SavedModel, last checkpoint, frozen inference graph and pipeline configuration.

_training_ folder will keep the checkpoints and evaluation data produced during retraining.

_inference-graph_ will be filled after exporting retrained object detection TensorFlow graph. It will contain inference graph, associated checkpoint files, a frozen inference graph and a SavedModel. Exporting for TensorFlow Lite will produce few more files.

### Prepare training dataset

To prepare the COCO data set for training run the following scripts. The process will take several hours as the amount of information to download, extract and preprocess is about 100 Gb.

Limited dataset [is configured](dataset-limited.sh#L8) to include 9 classes of common objects by default: person and all types of vehicles. To select other classes set environmental variable `CATEGORIES` to the list of classes separated with spaces prior to running `dataset-limited.sh`.

```bash
./dataset-complete.sh

export CATEGORIES="bear horse giraffe"      # optional
./dataset-limited.sh
```

### Download and prepare pretrained model

To download a pretrained object detection model and prepare it for training run:

```bash
python -m model -m ssd_mobilenet_v1_coco_2018_01_28 -d limited
```

To enable quantization aware training add `-q` flag:

```bash
python -m model -m ssd_mobilenet_v2_coco_2018_03_29 -d limited -q
```

The script configures a pipeline to train the neural network on a complete or limited dataset. The pipeline will then have references to the binary records and the label map prepared earlier.

`model.py` is set up to extract data from certain archives. Adapt the script to the archive structure if it cannot parse the loaded model.   

### Evaluate your model

Improving inference speed would only make sense if the accuracy of retrained model is not gotten worse. You may want to assess at first how accurate your model is by knowing its [Average Precision](https://cocodataset.org/#detection-eval).

```bash
python -m eval -m ssd_inception_v2_coco_2018_01_28
```

or 

```bash
./eval.sh
```

If the pipeline of a pretrained model is configured against a complete dataset, the average precision (AP) in the output will match the precision claimed for the given model.

The average precision got on a limited dataset is usually higher. It does not mean however that the model retrained will detect objects better. The architecture remains the same and so the actual accuracy. Retraining will take long, maybe weeks or even months. To know well ahead the target value of average precision when the model is fully retrained and you can stop training, run the same script on a pretrained model configured against a limited dataset.

### Train the model

Start training by running the following script and choosing a model:

```bash
./train.sh
```

Training takes much time, you can't keep the terminal session open all the time, so run training in the background:

```bash
export MODEL_DIR=$(pwd)/model/ssd_inception_v2_coco_2018_01_28 \
    && ./train.sh 1> output1.log 2> output2.log &
```

You can monitor the process like that then: 

```bash
tail output2.log -f
```

### Export inference graph

Having trained a model export an object detection TensorFlow graph for inference. Select the model and the last checkpoint. The result will consist of a frozen inference graph, associated checkpoint files and a SavedModel.

```bash
./export.sh
```

If the model has been trained using quantization aware training, it can be exported to TensorFlow Lite format:

```bash
./export-tflite.sh
```

Bear in mind that the format of the exported inference graph depends on the version of the API (see [Troubleshooting](#troubleshooting) section).

### Convert to various formats

The model exported as described above can be run using TensorFlow. To run the inference on the devices with hardware acceleration a conversion is needed. Converting models reduces their file size and introduces optimizations that do not affect accuracy.

To convert your model to TensorFlow Lite for running on mobile and other embedded devices do:

```bash
./convert-tflite.sh
```

To make it compatible for running on Nvidia CUDA GPUs using TensorRT, convert your model to UFF format:

```bash
python -m convert-uff -m ssd_inception_v2_coco_2018_01_28 -t -i
```

Only the models defined in `config/` folder can be converted to UFF.

## Troubleshooting

### 1. ValueError: ... no arg_scope defined for the base feature extractor

If training fails immediately after starting with the message above, open `pipeline.config` of the pretrained model and under `feature_extractor` right after `conv_hyperparameters` closing bracket paste the following line:
 
 ```
 override_base_feature_extractor_hyperparams: true
 ```

### 2. Building TRT engine fails for retrained UFF-model

The model can be successfully converted to UFF format but failed to run on TensorRT. This happens when the inference graph is exported using the version of object detection API other than that was  used to generate the pretrained graph downloaded from the web. The function calls and serialization format of inference graphs change from version to version. 

Checkout older commit `ae0a9409212d0072938fa60c9f85740bb89ced7e` of object detection API and [install](https://github.com/tensorflow/models/blob/ae0a9409212d0072938fa60c9f85740bb89ced7e/research/object_detection/g3doc/installation.md) it. Then export and convert the model again. 

## License

[MIT License](LICENSE)