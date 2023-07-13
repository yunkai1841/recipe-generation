# recipe-generation

NLP Text generation task. Generate recipe from ingredients.


## Setup
1. Download models and LLaMA-Adapter.
Use [setup.sh](setup.sh).
2. Prepare dataset. See [Prepare dataset](#prepare-dataset).

## Dataset
- [recipe-nlg](https://recipenlg.cs.put.poznan.pl/)

**NOTE** This dataset is only for non-commercial research and educational purposes. 
You have to agree to the terms of use to download the dataset.

### Prepare dataset
1. Put `full_dataset.csv` in `data/`
2. Run script to convert, analyze and make subset of dataset.
```bash
python script/convert_recipenlg.py
python script/summary_dataset.py
python script/subset_dataset.py
```

## Train
1. Install requirements for [LLaMA-Adapter](LLaMA-Adapter/README.md).
2. Run [train.sh](train.sh) to train model.  
Spec: RTX A6000, 20 min for 1k train dataset
3. Run
[LLaMA-Adapter/alpaca_finetuning_v1/extract_adapter_from_checkpoint.py](LLaMA-Adapter/alpaca_finetuning_v1/extract_adapter_from_checkpoint.py)
to extract adapter from checkpoint.

**NOTE** 30GB GPU memory is required.

**NOTE2** Windows is not supported.

## Inference
1. Install requirements for [LLaMA-Adapter](LLaMA-Adapter/README.md).
2. Put adapter model at `adapter-model/recipe_adapter_len10_layer30_epoch5.pth`
3. Run [run_app.sh](run_app.sh) to run web app.  
Spec: RTX A6000, 20 sec / 1 inference

**NOTE** 22GB GPU memory is required.
