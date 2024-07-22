#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/glm-4-9b-chat"

CUDA_VISIBLE_DEVICES=0 python webui_glm4.py \
  --model $MODEL_DIR
