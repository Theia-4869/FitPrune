#!/bin/bash

CKPT="llava-v1.5-7b"
METHOD="fitprune"
R=${1}
PARAM="R_${R}"

python -m llava.eval.model_vqa_loader \
    --model-path /mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/huggingface/${CKPT} \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --use-fitprune \
    --reduction-ratio ${R} \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${CKPT}/${METHOD}/${PARAM}

cd eval_tool

python calculation.py --results_dir answers/${CKPT}/${METHOD}/${PARAM}
