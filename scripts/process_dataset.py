# %%
from datasets import load_dataset

# the dataset needs to be manually downloaded from
# https://recipenlg.cs.put.poznan.pl/
#
dataset = load_dataset(
    "recipe_nlg", data_dir="E:\\dataset\\recipenlg", cache_dir="E:\\dataset\\recipenlg"
)

# %%
# print the dataset info
print(dataset)
# %%
# print the first 5 examples
print(dataset["train"][:5])
