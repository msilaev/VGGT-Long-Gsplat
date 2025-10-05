#!/bin/bash
set -e  # Exit on error

WORKDIR="experiments"

echo "--WORKDIR: $WORKDIR"

COLMAP_WORKSPACE="$WORKDIR/IMAGES_DIR_vgg_long_colmap/images"
UNDISTORTED_COLMAP_OUTPUT="${COLMAP_WORKSPACE}_undistorted"
IMAGE_DIR="${COLMAP_WORKSPACE}"
TRAIN_SCRIPT="train.py"

#echo "--- Starting Gaussian Splatting Pipeline ---"
echo "Input workspace: $COLMAP_WORKSPACE"
echo "Undistorted output: $UNDISTORTED_COLMAP_OUTPUT"

# --- Step 3: Undistort Images ---
echo "3. Undistorting COLMAP output..."
mkdir -p "$UNDISTORTED_COLMAP_OUTPUT"

echo "Image dir: $IMAGE_DIR"

colmap image_undistorter \
    --image_path "$IMAGE_DIR" \
    --input_path "$IMAGE_DIR/sparse" \
    --output_path "$UNDISTORTED_COLMAP_OUTPUT" \
    --output_type COLMAP \
    --max_image_size 2000

# --- Copy DB and Convert Sparse Model ---
#cp "$IMAGE_DIR/database.db" "$UNDISTORTED_COLMAP_OUTPUT/"

mkdir -p "$UNDISTORTED_COLMAP_OUTPUT/sparse/0_txt"

echo "3. Convert COLMAP output to txt"
colmap model_converter \
    --input_path "$UNDISTORTED_COLMAP_OUTPUT/sparse" \
    --output_path "$UNDISTORTED_COLMAP_OUTPUT/sparse/0_txt" \
    --output_type TXT

echo "Undistortion complete."

