import json
import csv
import numpy as np
import argparse


dataset_path = "data/recipe_nlg.json"
target_dataset_path = "data/recipe_nlg_subset.json"
summary_file = "data/summary.csv"

def subset_dataset(maxtokens: int, choosenum: int, random: bool = True):
    print("loading...")
    csv_data = []
    with open(summary_file, "r") as f:
        reader = csv.reader(f)
        # header: index,instruction,input,output,sums
        next(reader)
        for row in reader:
            csv_data.append(row)
    csv_data = np.array(csv_data)
    csv_data = csv_data.astype(np.int32)
    # filter by maxtokens
    csv_data = csv_data[csv_data[:, 4] <= maxtokens]
    print(f"number of available data: {len(csv_data)}")

    # choose data
    print("choosing data...")
    if random:
        choosenum = min(choosenum, len(csv_data))
        choosenum = max(choosenum, 1)
        choosenum = np.random.choice(len(csv_data), choosenum, replace=False)
        choosenum = np.sort(choosenum)
        csv_data = csv_data[choosenum]
    else:
        choosenum = min(choosenum, len(csv_data))
        choosenum = max(choosenum, 1)
        csv_data = csv_data[:choosenum]

    # save dataset
    print("saving dataset...")
    subset = []
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    for i in csv_data[:, 0]:
        subset.append(dataset[i])
    with open(target_dataset_path, "w") as f:
        json.dump(subset, f, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxtokens", type=int, default=512)
    parser.add_argument("--choosenum", type=int, default=1000)
    args = parser.parse_args()
    subset_dataset(args.maxtokens, args.choosenum)