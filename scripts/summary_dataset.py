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
tqdm_bar = tqdm.tqdm(total=len(dataset))
summary = []
for i, d in enumerate(dataset):
    instruction = tokenizer.encode(d["instruction"], bos=False, eos=False)
    input = tokenizer.encode(d["input"], bos=False, eos=False)
    output = tokenizer.encode(d["output"], bos=False, eos=False)
    summary.append(
        [
            i,
            len(instruction),
            len(input),
            len(output),
            len(instruction) + len(input) + len(output),
        ]
    )
    tqdm_bar.update(1)
tqdm_bar.close()


# %%
# save to csv
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "instruction", "input", "output", "sum"])
    writer.writerows(summary)

# %%
# show graph
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(output_file)
# sumの分布を表示 対数グラフ
df["sum"].hist(bins=30, log=True)
# 横軸：sumの値, 縦軸：その値をとるデータの数
# label表示
plt.xlabel("sum of tokens")
plt.ylabel("number of data")
plt.savefig("data/sum.png")
# plt.show()

# %%
# outputの分布を表示
plt.clf()
df["output"].hist(bins=30, log=True)
plt.xlabel("number of tokens in output")
plt.ylabel("number of data")
plt.savefig("data/output.png")
# plt.show()
# %%
