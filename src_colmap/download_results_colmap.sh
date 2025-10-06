#!/bin/bash
set -e


# Load environment variables
if [ -f ".env" ]; then
  source .env
fi

WORKDIR="/home/GAUSSIAN-SPLATTING"

GSPLAT_OUTPUT_DIR="GSPLAT_OUTPUT_DIR_colmap"
IMAGE_DIR="bonsai"
IMAGE_DIR_REMOTE="IMAGES_DIR_colmap"
SAVE_DIR="/worktmp/GAUSSIAN-SPLATTING/experiments/RESULTS/COLMAP/$IMAGE_DIR/"


rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

rsync -avz "$REMOTE_USER@$REMOTE_HOST:$WORKDIR/gsplat/examples/results/$GSPLAT_OUTPUT_DIR/ply/point_cloud_29999.ply" $SAVE_DIR
rsync -avz "$REMOTE_USER@$REMOTE_HOST:$WORKDIR/gsplat/examples/results/$GSPLAT_OUTPUT_DIR/videos" $SAVE_DIR

rsync -avz "$REMOTE_USER@$REMOTE_HOST:$WORKDIR/gsplat/examples/results/$GSPLAT_OUTPUT_DIR/videos" $SAVE_DIR


WORKDIR="/home/GAUSSIAN-SPLATTING/experiments"
rsync -avz "$REMOTE_USER@$REMOTE_HOST:$WORKDIR/$IMAGE_DIR_REMOTE/pipeline.log" "$SAVE_DIR/"

rsync -avz "$REMOTE_USER@$REMOTE_HOST:$WORKDIR/$IMAGE_DIR_REMOTE/images_undistorted/sparse" "$SAVE_DIR"

echo "[LOCAL] Pipeline complete."

