#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python ../Train_PAT_Charades.py \
-dataset charades \
-mode rgb \
-model PAT \
-train True \
-rgb_root ./data/Charades_RGB \
-num_clips 256 \
-skip 0 \
-comp_info False \
-epochs 30 \
-unisize True \
-batch_size 5 \
-num_classes 157 \
-lr 0.0001 \
-annotation_file ./data/charades.json \
-step_size 7 \
-gamma 0.1 \

