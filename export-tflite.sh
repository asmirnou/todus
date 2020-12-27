#!/usr/bin/env bash

set -e

source util.sh
select_model
select_trained_checkpoint
cd $(which_object_detection)

python object_detection/export_tflite_ssd_graph.py \
    --input_type="image_tensor" \
    --add_postprocessing_op=true \
    --pipeline_config_path="$MODEL_DIR/pretrained/pipeline.config" \
    --trained_checkpoint_prefix="$MODEL_DIR/training/$CHECKPOINT_PREFIX" \
    --output_directory="$MODEL_DIR/inference-graph"
