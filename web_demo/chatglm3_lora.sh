#!/bin/bash
MODEL_DIR="/root/autodl-tmp/chatglm3-6b"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/hotel-chatglm3-lora"

CUDA_VISIBLE_DEVICES=0 python webui_chatglm3.py \
  --model_name_or_path $MODEL_DIR \
  --checkpoint_path $CHECKPOINT_DIR \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1
