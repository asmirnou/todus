#!/usr/bin/env bash

function which_object_detection {
    python object-detection.py
}

function select_model {
    if [[ -n "$MODEL_DIR" ]]; then
        return
    fi

    CUR_DIR=$(pwd)
    cd model
    printf "Please select object detection model:\n"
    select d in */; do test -n "$d" && break; echo ">>> Invalid Selection"; done
    cd $d
    MODEL_DIR=$(pwd)
    printf "Selected model \"$d\"\n"
    cd $CUR_DIR
}

function select_trained_checkpoint {
    if [[ -n "$CHECKPOINT_PREFIX" ]]; then
        return
    fi

    CUR_DIR=$(pwd)
    cd $MODEL_DIR/training/
    printf "Please select trained checkpoint:\n"
    select d in model.ckpt-*.index; do test -n "$d" && break; echo ">>> Invalid Selection"; done
    CHECKPOINT_PREFIX=${d%.index}
    printf "Selected trained checkpoint prefix \"$CHECKPOINT_PREFIX\"\n"
    cd $CUR_DIR
}
