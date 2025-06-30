#!/bin/bash

cd ./weights

# VGGT
wget https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt

# SALAD
FILE_ID="1u83Dmqmm1-uikOPr58IIhfIzDYwFxCy1"
FILENAME="downloaded_file"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILE_ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" -O $FILENAME && rm -rf /tmp/cookies.txt

# DINO
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# DBoW
wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz
tar -xzvf ORBvoc.txt.tar.gz
rm ORBvoc.txt.tar.gz