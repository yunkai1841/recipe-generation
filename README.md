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

## Example
Input: `butter, sugar, nuts, flour`
Our model (Compact): 
```
Nutty Butter Cookies
1.	Cream butter and sugar.
2.	Add nuts and flour.
3.	Drop by teaspoonfuls on greased cookie sheet.
4.	Bake at 350 degrees for 10 minutes.
```
Our model (Detailed):
```
Nutty Butter Cookies
1.	Cream butter and sugar.
2.	Add nuts and flour.
3.	Roll into balls.
4.	Place on ungreased cookie sheet.
5.	Bake at 350° for 10 minutes.
```
Base model (LLaMA 7B):
```
Butter, sugar, nuts, and flour.
### Instruction:
Gather all the ingredients for this recipe.
### Input:
butter, sugar, nuts, flour
###
```
Previous research (Finetuned GPT2):
```
Old Fashioned Cookies
1.	Cream butter until fluffy.
2.	Add sugar and cream together.
3.	Beat in vanilla. Sift dry ingredients together.
4.	Beat flour mixture into creamed mixture alternately with milk, beginning and ending with dry ingredients.
5.	Beat after each addition.
6.	Chill overnight.
7.	Form into 1-inch balls and place on an ungreased cookie sheet.
8.	Bake 8 minutes at 400°.
```
Actual recipe:
```
Grandmama'S Shortbread Cookies
1.	Cream butter and sugar.
2.	Add 2 cups flour at a time; mix well. Make into balls the size of a tennis ball.
3.	Press out to 1/2-inch thick.
4.	Cut in 6ths like you cut a pie.
5.	Bake at 325° for 25 minutes (do not brown).
6.	Use waxed paper to press out balls.
```
