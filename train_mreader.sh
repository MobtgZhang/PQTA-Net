#!/bin/bash
export PYTHONIOENCODING=utf-8

unset CUDA_VISIBLE_DEVICES

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m paddle.distributed.launch --gpus "0" run.py \
    --do_train \
    --device gpu \
    --model-type mreader \
