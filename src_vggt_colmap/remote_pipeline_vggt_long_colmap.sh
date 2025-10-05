#!/bin/bash
set -e
set -x

REMOTE_IMAGE_DIR="$1"
if [ -z "$REMOTE_IMAGE_DIR" ]; then
  echo "Usage: remote_pipeline.sh REMOTE_IMAGE_DIR"
  exit 1
fi

WORKDIR="/home/hdd/mikhail/GAUSSIAN-SPLATTING/experiments"
GSPLAT_OUTPUT_DIR="GSPLAT_OUTPUT_DIR_vggtlong"

cd "$REMOTE_IMAGE_DIR"

# Load Conda into this shell session
source  /conda.sh

# Step 2: Colmap Reconstruction

conda deactivate
conda activate vggsfm_tmp
#conda install xformers -c xformers

cd VGGT-Long-Gsplat
rm -rf exps

time_start=$(date +%s)

echo "[REMOTE] Running vggt long..."
python vggt_long.py --image_dir "$REMOTE_IMAGE_DIR" || {
  echo "[ERROR] vggt_long.py failed!"
  exit 1
}
echo "[REMOTE] vggt long finished successfully"


echo "[REMOTE] Running demo_colmap..."
python demo_colmap.py --scene_dir "$REMOTE_IMAGE_DIR" --max_reproj_error 16 --use_ba --max_query_pts=1024 --query_frame_num=6

time_end=$(date +%s)
time_diff=$((time_end - time_start))
echo "Sparse reconstruction took ${time_diff} sec"

conda deactivate
conda activate colmap_env

# Step 3: Undistort
echo "[REMOTE] Running COLMAP undistortion..."

cd "$WORKDIR"
chmod +x undistorting_script_vggt_long_colmap.sh
./undistorting_script_vggt_long_colmap.sh

# Step 4: Prepare gsplat
echo "[REMOTE] Preparing gsplat..."

# Step 5: Copy images
echo "[REMOTE] Duplicating image directory for training..."
cp -r "${REMOTE_IMAGE_DIR}_undistorted/images" "${REMOTE_IMAGE_DIR}_undistorted/images_1"

# Step 6: Training
conda deactivate
conda activate py11

echo "[REMOTE] Starting gsplat training..."
cd "$WORKDIR/../gsplat/examples"
mkdir -p "$WORKDIR/../gsplat/examples/results/$GSPLAT_OUTPUT_DIR"

#mkdir -p "${REMOTE_IMAGE_DIR}_undistorted/sparse/0"
#mv ${REMOTE_IMAGE_DIR}_undistorted/sparse/*.bin "${REMOTE_IMAGE_DIR}_undistorted/sparse/0/"

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir "${REMOTE_IMAGE_DIR}_undistorted" \
    --data_factor 1 \
    --result_dir "$WORKDIR/../gsplat/examples/results/$GSPLAT_OUTPUT_DIR" \
    --save_ply \
    --ply_steps 30000 \
    --disable_viewer \
    --render_traj_path "ellipse"

echo "[REMOTE] Training complete."

