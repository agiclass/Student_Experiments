#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/Qwen2-7B-Instruct"
CHECKPOINT_DIR="/root/checkpoints/hotel-qwen2-lora"

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --data ../data/test.jsonl
