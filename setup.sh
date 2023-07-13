#!/bin/bash

# clone LLaMA-Adapter
git clone https://github.com/OpenGVLab/LLaMA-Adapter.git

# download LLaMA base model
# LLaMA-base/
# ├── 7B
# │   ├── consolidated.00.pth
# │   └── params.json
# └── tokenizer.model
mkdir -p LLaMA-base/7B
wget https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/consolidated.00.pth -P LLaMA-base/7B
wget https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/params.json -P LLaMA-base/7B
wget https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/tokenizer.model -P LLaMA-base

# if input is y (yes), install requirements
read -p "Install requirements? (y/n): " yn
case "$yn" in
    [yY]*) pip install -r requirements.txt;;
    *) echo "Skip installing requirements.";;
esac
