# recipe-generation
![Static Badge](https://img.shields.io/badge/under%20implementation-orange?style=for-the-badge)

NLP Text generation task. Generate recipe by fine tuned LLaMA model. 


## Dataset
- [recipe-nlg](https://recipenlg.cs.put.poznan.pl/)

**NOTE** This dataset is only for non-commercial research and educational purposes. 
You have to agree to the terms of use to download the dataset.

## Model
Base model is LLaMA-7B.
We fine tuned LLaMA-7B model with recipe-nlg dataset.

## Setup
You need to download models and LLaMA-Adapter first.
Use [setup.sh](setup.sh).

## How to train
You need to install requirements for [LLaMA-Adapter](LLaMA-Adapter/README.md) first.

**NOTE** Windows is not supported.

