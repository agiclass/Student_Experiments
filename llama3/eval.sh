#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/meta-llama-3.1-8b-instruct"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/hotel-llama3-qlora"

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --data ../data/test.jsonl
