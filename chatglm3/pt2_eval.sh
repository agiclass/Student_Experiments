#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/chatglm3-6b"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/hotel-chatglm3-pt2"

CUDA_VISIBLE_DEVICES=0 python cli_evaluate.py \
  --test_file ../data/test.chatglm3.jsonl \
  --model_name_or_path $MODEL_DIR \
  --checkpoint_path $CHECKPOINT_DIR \
  --pre_seq_len 256
