#!/bin/bash

CKPT="llava-v1.5-7b"

# python llava/eval/extract_subset.py \
#     --src ./playground/data/train/llava_v1_5_mix665k.json \
#     --dst ./playground/data/train/llava_v1_5_mix665.jsonl \
#     --ratio 0.001\
#     --seed 2024

CUDA_VISIBLE_DEVICES=0 python llava/eval/statistical_analysis.py \
    --model-path /mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/huggingface/${CKPT} \
    --image-folder ./playground/data/train \
    --question-file ./playground/data/train/llava_v1_5_mix665.jsonl \
    --conv-mode vicuna_v1 \
    --reduction_ratio 50
