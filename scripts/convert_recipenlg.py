from datasets import load_dataset
import tqdm
import json

# the dataset needs to be manually downloaded from
# https://recipenlg.cs.put.poznan.pl/
# replace the path below with the path to the dataset
dataset_dir = "data"
dataset = load_dataset("recipe_nlg", data_dir=dataset_dir)

# print the dataset info
print(dataset)

# format the dataset into alpaca format

instruction_format = "Generate a recipe with these ingredients."

json_data = []
tqdm_bar = tqdm.tqdm(total=len(dataset["train"]))
for data in dataset["train"]:
    instruction = instruction_format
    output = data["title"] + "\n"
    for i, direction in enumerate(data["directions"]):
        output += f"{i+1}. {direction}\n"
    input = ""
    for ner in data["ner"]:
        input += ner + ", "
    input = input[:-2]

    json_data.append(
        {
            "instruction": instruction,
            "input": input,
            "output": output,
        }
    )
    tqdm_bar.update(1)

tqdm_bar.close()

# save the dataset
print("Saving the dataset...")
with open("data/recipe_nlg.json", "w") as f:
    json.dump(json_data, f, indent=4)
