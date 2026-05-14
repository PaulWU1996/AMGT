#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python ../Train_PAT_MultiTHUMOS.py \
-dataset charades \
-mode rgb \
-model PAT \
-train True \
-rgb_root Path_to_data \
-num_clips 256 \
-skip 0 \
-comp_info False \
-epochs 50 \
-unisize True \
-batch_size 1 \
-num_classes 65 \
-lr 0.0001 \
-annotation_file ./data/multhithumos.json \
-step_size 35 \
-gamma 0.1 \

