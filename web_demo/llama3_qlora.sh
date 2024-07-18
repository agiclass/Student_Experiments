MODEL_DIR="/root/autodl-tmp/Meta-Llama-3-8B-Instruct"
CHECKPOINT_DIR="/root/checkpoints/hotel-llama3-qlora"

CUDA_VISIBLE_DEVICES=0 python webui_llama3.py \
    --model $MODEL_DIR \
    --ckpt $CHECKPOINT_DIR
