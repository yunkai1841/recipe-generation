import json
import csv
import numpy as np
import argparse


dataset_path = "data/recipe_nlg.json"
target_dataset_path = "data/recipe_nlg_subset.json"
test_dataset_path = "data/recipe_nlg_test.json"
summary_file = "data/summary.csv"


def subset_dataset(
    maxtokens: int,
    subsetnum: int,
    random: bool = True,
    savetest: bool = False,
    testnum: int = 100,
):
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
    if savetest:
        targetnum = min(subsetnum + testnum, len(csv_data))
    else:
        targetnum = min(subsetnum, len(csv_data))
    targetnum = max(targetnum, 1)
    if random:
        target = np.random.choice(len(csv_data), targetnum, replace=False)
        csv_data = csv_data[target]
    else:
        csv_data = csv_data[:targetnum]

    # save dataset
    print("saving dataset...")
    subset = []
    testset = []
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    for i in csv_data[:subsetnum, 0]:
        subset.append(dataset[i])
    with open(target_dataset_path, "w") as f:
        json.dump(subset, f, indent=4)
    if savetest:
        for i in csv_data[-testnum:, 0]:
            testset.append(dataset[i])
        with open(test_dataset_path, "w") as f:
            json.dump(testset, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Make subset of dataset"
    parser.add_argument(
        "--maxtokens",
        "-m",
        type=int,
        default=100,
        help="max number of tokens for each data",
    )
    parser.add_argument(
        "--subsetnum", "-n", type=int, default=1000, help="number of subset data"
    )
    parser.add_argument(
        "--savetest", "-t", action="store_true", help="save test dataset"
    )
    parser.add_argument(
        "--testnum", "-tn", type=int, default=100, help="number of test data"
    )
    args = parser.parse_args()
    subset_dataset(
        args.maxtokens, args.subsetnum, savetest=args.savetest, testnum=args.testnum
    )
