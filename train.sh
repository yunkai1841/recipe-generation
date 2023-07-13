#!/bin/bash
cd LLaMA-Adapter/alpaca_finetuning_v1

TARGET_FOLDER=../../LLaMA-base
DATA_PATH=../../data/recipe_nlg_subset.json

torchrun --nproc_per_node 1 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --data_path $DATA_PATH \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size 4 \
    --epochs 5 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./checkpoint/
