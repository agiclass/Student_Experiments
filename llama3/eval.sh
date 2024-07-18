#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/Meta-Llama-3-8B-Instruct"
CHECKPOINT_DIR="/root/checkpoints/hotel-llama3-qlora"

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --data ../data/test.jsonl
