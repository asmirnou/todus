#!/usr/bin/env bash

set -e

SCRIPTDIR=$(pwd)

if [[ -z "CATEGORIES" ]]; then
    CATEGORIES="person bicycle car motorcycle airplane bus train truck boat"
fi
echo "Categories: $CATEGORIES"

SCRATCH_DIR="${SCRIPTDIR}/dataset/mscoco/raw-data"
TRAIN_IMAGE_DIR="${SCRATCH_DIR}/train2017"
VAL_IMAGE_DIR="${SCRATCH_DIR}/val2017"
TEST_IMAGE_DIR="${SCRATCH_DIR}/test2017"
TRAIN_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/instances_train2017.json"
VAL_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/instances_val2017.json"
TEST_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/image_info_test2017.json"
TESTDEV_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/image_info_test-dev2017.json"

cd ${SCRATCH_DIR}

if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -oq"
fi

echo "Unzipping annotations"
${UNZIP} "$SCRATCH_DIR/annotations_trainval2017.zip"
${UNZIP} "$SCRATCH_DIR/image_info_test2017.zip"

cd ${SCRIPTDIR}

python filter-coco.py \
    --input_json  "$TRAIN_ANNOTATIONS_FILE" \
    --output_json "$TRAIN_ANNOTATIONS_FILE" \
    --categories $CATEGORIES

python filter-coco.py \
    --input_json  "$VAL_ANNOTATIONS_FILE" \
    --output_json "$VAL_ANNOTATIONS_FILE" \
    --categories $CATEGORIES

python filter-coco.py \
    --input_json  "$TEST_ANNOTATIONS_FILE" \
    --output_json "$TEST_ANNOTATIONS_FILE" \
    --categories $CATEGORIES

python filter-coco.py \
    --input_json  "$TESTDEV_ANNOTATIONS_FILE" \
    --output_json "$TESTDEV_ANNOTATIONS_FILE" \
    --categories $CATEGORIES

source untitled/util.sh
cd $(which_object_detection)

python object_detection/dataset_tools/create_coco_tf_record.py \
  --logtostderr \
  --include_masks \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --test_image_dir="${TEST_IMAGE_DIR}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
  --output_dir="$SCRIPTDIR/dataset/mscoco"

cp object_detection/data/mscoco_complete_label_map.pbtxt \
    "$SCRIPTDIR/dataset/mscoco/mscoco_label_map.pbtxt"
    
cd $SCRIPTDIR
python filter-label.py \
    --input_pdtxt "$SCRIPTDIR/dataset/mscoco/mscoco_label_map.pbtxt" \
    --output_pdtxt "$SCRIPTDIR/dataset/mscoco/mscoco_label_map.pbtxt" \
    --categories $CATEGORIES

cd "$SCRIPTDIR/dataset/mscoco"
set +e
mkdir limited
mv coco* limited
mv mscoco* limited
