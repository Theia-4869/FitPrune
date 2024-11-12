#!/bin/bash

CKPT="llava-v1.5-7b"
METHOD="fitprune"
R=${1}
PARAM="R_${R}"

python -m llava.eval.model_vqa_mmbench \
    --model-path /mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/huggingface/${CKPT} \
    --question-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --single-pred-prompt \
    --use-fitprune \
    --reduction-ratio ${R} \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/${CKPT}/${METHOD} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/${CKPT}/${METHOD} \
    --experiment ${PARAM}
