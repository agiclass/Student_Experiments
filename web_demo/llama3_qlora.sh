MODEL_DIR="/root/autodl-tmp/meta-llama-3.1-8b-instruct"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/hotel-llama3-qlora"

CUDA_VISIBLE_DEVICES=0 python webui_llama3.py \
    --model $MODEL_DIR \
    --ckpt $CHECKPOINT_DIR
