torchrun --nproc_per_node 1 inference/app.py \
    --ckpt_dir ./LLaMA-base/7B \
    --tokenizer_path ./LLaMA-base/tokenizer.model \
    --adapter_path ./adapter-model/recipe_adapter_len10_layer30_epoch5.pth \
    --share_gradio True
