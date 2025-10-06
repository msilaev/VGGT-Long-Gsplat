#!/bin/bash
set -e
set -x

REMOTE_IMAGE_DIR="$1"
if [ -z "$REMOTE_IMAGE_DIR" ]; then
  echo "Usage: remote_pipeline.sh REMOTE_IMAGE_DIR"
  exit 1
fi

WORKDIR="/home/GAUSSIAN-SPLATTING/experiments"
GSPLAT_OUTPUT_DIR="GSPLAT_OUTPUT_DIR_colmap"

cd "$REMOTE_IMAGE_DIR"

# Load Conda into this shell session
source /home/miniconda3/etc/profile.d/conda.sh

# Step 2: Colmap Reconstruction

conda deactivate
conda activate colmap_env

echo "[REMOTE] Running COLMAP reconstruction..."
time_start=$(date +%s)
colmap automatic_reconstructor --workspace_path . --image_path "$REMOTE_IMAGE_DIR" --dense 0 > colmap.log 2>&1 &
wait

time_end=$(date +%s)
time_diff=$((time_end - time_start))
echo "Sparse reconstruction took ${time_diff} sec"

# Step 3: Undistort
echo "[REMOTE] Running COLMAP undistortion..."
cd "$WORKDIR"
chmod +x undistorting_script_colmap.sh
./undistorting_script_colmap.sh

# Step 4: Prepare gsplat
echo "[REMOTE] Preparing gsplat..."
#conda create -n py11 python=3.11 -y
#source activate py11

# Step 5: Copy images
echo "[REMOTE] Duplicating image directory for training..."
cp -r "${REMOTE_IMAGE_DIR}_undistorted/images" "${REMOTE_IMAGE_DIR}_undistorted/images_1"

# Step 6: Training
conda deactivate
conda activate py11

echo "[REMOTE] Starting gsplat training..."
cd "$WORKDIR/../gsplat/examples"
mkdir -p "$WORKDIR/../gsplat/examples/results/$GSPLAT_OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir "${REMOTE_IMAGE_DIR}_undistorted" \
    --data_factor 1 \
    --result_dir "$WORKDIR/../gsplat/examples/results/$GSPLAT_OUTPUT_DIR" \
    --save_ply \
    --ply_steps 30000 \
    --disable_viewer \
    --render_traj_path "ellipse"

echo "[REMOTE] Training complete."

