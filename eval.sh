#!/usr/bin/env bash

set -e

source util.sh
select_model
cd $(which_object_detection)

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/model_main.py \
    --run_once=true \
    --model_dir="$MODEL_DIR/training" \
    --eval_dir="$MODEL_DIR/training" \
    --checkpoint_dir="$MODEL_DIR/training" \
    --pipeline_config_path="$MODEL_DIR/pretrained/pipeline.config"
