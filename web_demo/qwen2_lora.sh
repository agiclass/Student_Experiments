MODEL_DIR="/root/autodl-tmp/Qwen2-7B-Instruct"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/hotel-qwen2-lora"

CUDA_VISIBLE_DEVICES=0 python webui_qwen2.py \
    --model $MODEL_DIR \
    --ckpt $CHECKPOINT_DIR
