# load the dataset
import json
import csv
from tqdm import tqdm
from llama.tokenizer import Tokenizer
from concurrent.futures import ThreadPoolExecutor

tokenizer = Tokenizer("LLaMA-base/tokenizer.model")
dataset_path = "data/recipe_nlg.json"
output_file = "data/summary.csv"

with open(dataset_path, "r") as f:
    dataset = json.load(f)


# summarize the length of tokens
def calculate_token_lengths(i, d):
    instruction = len(tokenizer.encode(d["instruction"], bos=False, eos=False))
    input = len(tokenizer.encode(d["input"], bos=False, eos=False))
    output = len(tokenizer.encode(d["output"], bos=False, eos=False))
    return i, instruction, input, output, instruction + input + output


summary = []
with ThreadPoolExecutor() as executor:
    futures = []
    print("calculating tokens...")
    tqdm_bar = tqdm(total=len(dataset))
    for i, d in enumerate(dataset):
        exe = executor.submit(calculate_token_lengths, i, d)
        # update progress bar
        exe.add_done_callback(lambda p: tqdm_bar.update(1))
        futures.append(exe)

    tqdm_bar.close()
    print("collecting results...")
    for future in futures:
        summary.append(future.result())

summary.sort(key=lambda x: x[0])

# save to csv
print("saving summary...")
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "instruction", "input", "output", "sum"])
    writer.writerows(summary)

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

# outputの分布を表示
plt.clf()
df["output"].hist(bins=30, log=True)
plt.xlabel("number of tokens in output")
plt.ylabel("number of data")
plt.savefig("data/output.png")
# plt.show()
