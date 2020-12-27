#!/usr/bin/env bash

set -e

source util.sh
select_model

tflite_convert \
  --output_file="$MODEL_DIR/ssd_coco_quant_postprocess.tflite" \
  --graph_def_file="$MODEL_DIR/inference-graph/tflite_graph.pb" \
  --input_arrays="normalized_input_image_tensor" \
  --output_arrays="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" \
  --input_shapes=1,300,300,3 \
  --allow_custom_ops \
  --inference_type=QUANTIZED_UINT8 \
  --mean_values=128 \
  --std_dev_values=128 \
  --change_concat_input_ranges=false \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true
#  --default_ranges_min=0 \
#  --default_ranges_max=255 \
#  --post_training_quantize
