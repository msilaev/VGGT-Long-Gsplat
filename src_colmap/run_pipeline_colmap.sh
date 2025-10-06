#!/bin/bash
set -e

REMOTE_DIR="/home/GAUSSIAN-SPLATTING/experiments"
IMAGE_DIR="IMAGES_DIR_colmap"

REMOTE_DIR_EXP="$REMOTE_DIR/$IMAGE_DIR"
REMOTE_IMAGE_DIR="$REMOTE_DIR/$IMAGE_DIR/images"

SCENE="bonsai"
LOCAL_SRC_DIR="/worktmp/GAUSSIAN-SPLATTING/VGGT-LONG-GSPLAT/src_colmap"
LOCAL_IMAGE_DIR="/worktmp/GAUSSIAN-SPLATTING/experiments/DATASETS/360_v2/$SCENE"

# Load environment variables
if [ -f ".env" ]; then
  source .env
fi

./crop_resize_local.sh "$LOCAL_IMAGE_DIR/images"

# Step 1: Copy files to remote machine
echo "[LOCAL] Syncing image directory to remote..."

# 1. Remove old remote directory
ssh "$REMOTE_USER@$REMOTE_HOST" "rm -rf '$REMOTE_DIR/$IMAGE_DIR'"
# 2. Create target directory
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_DIR/$IMAGE_DIR'"
# 3. Copy local folder contents (not the folder itself)
rsync -avz "$LOCAL_IMAGE_DIR"/ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/$IMAGE_DIR/"


# Step 2: Run remote pipeline via SSH
echo "[LOCAL] Starting remote processing pipeline..."
rsync -avz "$LOCAL_SRC_DIR/remote_pipeline_colmap.sh" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
rsync -avz "$LOCAL_SRC_DIR/undistorting_script_colmap.sh" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

ssh "$REMOTE_USER@$REMOTE_HOST" "nohup bash $REMOTE_DIR/remote_pipeline_colmap.sh $REMOTE_IMAGE_DIR > $REMOTE_DIR_EXP/pipeline.log 2>&1 &"

echo "[LOCAL] Pipeline complete."

