#! /usr/bin/env bash

set -ex

LR=2e-2
PRE_SEQ_LEN=256
MAX_SEQ_LEN=3072

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=hotel_pt2
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

BASE_MODEL_PATH=/root/autodl-tmp/chatglm3-6b

CUDA_VISIBLE_DEVICES=0 python main_pt2.py \
    --do_train \
    --do_eval \
    --train_file ../data/train.chatglm3.jsonl \
    --validation_file ../data/dev.chatglm3.jsonl \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --num_train_epochs 6 \
    --logging_steps 300 \
    --logging_dir $OUTPUT_DIR/logs \
    --save_steps 300 \
    --learning_rate $LR \
    --quantization_bit 4 \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
