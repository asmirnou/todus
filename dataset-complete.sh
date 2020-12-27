#!/usr/bin/env bash

set -e

SCRIPTDIR=$(pwd)

source util.sh
cd $(which_object_detection)

object_detection/dataset_tools/download_and_preprocess_mscoco.sh \
    "$SCRIPTDIR/dataset/mscoco"

cp object_detection/data/mscoco_complete_label_map.pbtxt \
    "$SCRIPTDIR/dataset/mscoco/mscoco_label_map.pbtxt"

cd "$SCRIPTDIR/dataset/mscoco"
set +e
mkdir complete
mv coco* complete
mv mscoco* complete
