#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --num_processes 4 --main_process_port 29600 train.py \
    --dataset_name expguardmix \
    --model_type qwen2.5-7b \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_seq_len 4096 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --epochs 3 \
    --seed 42 \
    --wandb_mode online \
    --attn_implementation flash_attention_2 \
    --do_train \
    --bf16 \
    --use_gradient_checkpointing
