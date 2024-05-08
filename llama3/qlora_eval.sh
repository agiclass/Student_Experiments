#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/Meta-Llama-3-8B-Instruct"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/hotel_qlora-llama3"

CUDA_VISIBLE_DEVICES=0 python cli_evaluate.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --data ../data/test.llama3.jsonl
