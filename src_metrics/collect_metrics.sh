#!/bin/bash
set -e

rm -f metrics.txt  # safer than rm -rf for a single file

# Loop through multiple image directories
for IMAGE_DIR in bonsai kitchen ignatius; do

    SAVE_DIR="/worktmp/GAUSSIAN-SPLATTING/experiments/RESULTS/COLMAP/$IMAGE_DIR/"

    echo "[LOCAL] Collecting metrics for $IMAGE_DIR..."

    # Run Python parser
    python collect_metrics.py \
        --log_file "$SAVE_DIR/pipeline.log" \
        --scene "$IMAGE_DIR" \
        --reconstructor_type COLMAP

done


# Loop through multiple image directories
for IMAGE_DIR in bonsai kitchen ignatius; do

    SAVE_DIR="/worktmp/GAUSSIAN-SPLATTING/experiments/RESULTS/VGGT_LONG/$IMAGE_DIR/"

    echo "[LOCAL] Collecting metrics for $IMAGE_DIR..."

    # Run Python parser
    python collect_metrics.py \
        --log_file "$SAVE_DIR/pipeline.log" \
        --scene "$IMAGE_DIR" \
        --reconstructor_type VGGT_LONG

done

echo "[LOCAL] Metrics collection complete."

echo "[LOCAL] Pipeline complete."

