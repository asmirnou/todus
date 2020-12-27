#!/usr/bin/env bash

set -e

source util.sh
select_model
cd $(which_object_detection)

python object_detection/model_main.py \
    --alsologtostderr \
    --model_dir="$MODEL_DIR/training" \
    --pipeline_config_path="$MODEL_DIR/pretrained/pipeline.config"
