#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/glm-4-9b-chat"
CHECKPOINT_DIR="/root/checkpoints/hotel-glm4-qlora"

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --data ../data/test.glm4.jsonl
