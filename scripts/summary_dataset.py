# %%
# load the dataset
import json
import csv
import tqdm
from llama.tokenizer import Tokenizer

# %%
tokenizer = Tokenizer("LLaMA-7B/tokenizer.model")
dataset_path = "data/recipe_nlg.json"
target_dataset_path = "data/recipe_nlg_subset.json"
output_file = "data/summary.csv"

with open(dataset_path, "r") as f:
    dataset = json.load(f)

# %%
# summarize the length of tokens
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "instruction", "input", "output"])
    tqdm_bar = tqdm.tqdm(total=len(dataset))
    for i, d in enumerate(dataset):
        instruction = tokenizer.encode(d["instruction"], bos=False, eos=False)
        input = tokenizer.encode(d["input"], bos=False, eos=False)
        output = tokenizer.encode(d["output"], bos=False, eos=False)
        writer.writerow([i, len(instruction), len(input), len(output)])
        tqdm_bar.update(1)
    tqdm_bar.close()
