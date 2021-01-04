#!/usr/bin/env bash

set -e

source util.sh
select_model
select_trained_checkpoint
cd $(which_object_detection)

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/export_inference_graph.py \
    --input_type="image_tensor" \
    --pipeline_config_path="$MODEL_DIR/pretrained/pipeline.config" \
    --trained_checkpoint_prefix="$MODEL_DIR/training/$CHECKPOINT_PREFIX" \
    --output_directory="$MODEL_DIR/inference-graph"

cp -n "$MODEL_DIR/pretrained/pipeline.config" "$MODEL_DIR/inference-graph"
