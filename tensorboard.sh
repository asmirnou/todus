#!/usr/bin/env bash

set -e

source util.sh
select_model

tensorboard --logdir="$MODEL_DIR/training"
