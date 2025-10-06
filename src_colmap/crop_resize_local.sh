#!/bin/bash
set -e

IMAGE_DIR="IMAGES_DIR_vgg_long_colmap"
#IMAGE_DIR="IMAGES_DIR_vgg"

LOCAL_IMAGE_DIR="/worktmp/GAUSSIAN-SPLATTING/experiments/IMAGES_DIR_vgg_long_colmap/images"  #"$1"

LOCAL_IMAGE_DIR="$1"
cd "$LOCAL_IMAGE_DIR"


shopt -s nullglob
for f in *.JPG; do
    mv "$f" "${f%.JPG}.jpg"
done
shopt -u nullglob



