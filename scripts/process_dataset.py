# %%
from datasets import load_dataset

# the dataset needs to be manually downloaded from
# https://recipenlg.cs.put.poznan.pl/
# replace the path below with the path to the dataset
dataset_dir = ""
dataset = load_dataset("recipe_nlg", data_dir=dataset_dir)

# %%
# print the dataset info
print(dataset)
# %%
# print the first 5 examples
print(dataset["train"][:5])
