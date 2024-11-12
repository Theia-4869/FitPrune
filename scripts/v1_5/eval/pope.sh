#!/bin/bash

CKPT="llava-v1.5-7b"
METHOD="fitprune"
R=${1}
PARAM="R_${R}"

python -m llava.eval.model_vqa_loader \
    --model-path /mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/huggingface/${CKPT} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --use-fitprune \
    --reduction-ratio ${R} \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${CKPT}/${METHOD}/${PARAM}.jsonl
