import pandas as pd
import os
from src.utils import get_project_root
import datetime
import json
from collections import defaultdict
import json
import sys

print(sys.argv[1])
with open(sys.argv[1]) as f:
    args_dict = json.load(f)

# SET THESE EACH RUN AS NEEDED
vocab_name = args_dict["vocab_name"]
diverse_vocab = args_dict.get("diverse_vocab", None)
diverse_vocab_level = args_dict.get("diverse_vocab_level", None)
vocab_cutoff_threshold = args_dict.get("vocab_cutoff_threshold", None)

download_run_name = args_dict["download_run_name"]
dataset_name = args_dict["dataset_name"]


genomes_folder = os.path.join(get_project_root(), "data", download_run_name)
dataset_folder = os.path.join(
    get_project_root(), "data", download_run_name, dataset_name
)
tokenized_folder = os.path.join(genomes_folder, "tokenized")
vocab_path = os.path.join(
    dataset_folder,
    f"{vocab_name}_vocab.json",
)
vocab_counts_path = os.path.join(
    dataset_folder,
    f"{vocab_name}_vocab_counts.csv",
)

start_time = datetime.datetime.now()
print(start_time)

tokenized_file = os.path.join(
    dataset_folder,
    f"train.csv",
)

if diverse_vocab == "True":
    tokens_occ_dict = defaultdict(int)
    tokens_occ_diverse_list_dict = defaultdict(set)
    diverse_index = None
    with open(tokenized_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                diverse_index = line.split(",").index(diverse_vocab_level)
                continue
            tokens = line.split(",")[0].split()
            taxon = line.split(",")[diverse_index]
            tokens = list(set(tokens))
            for token in tokens:
                if taxon not in tokens_occ_diverse_list_dict[token]:
                    tokens_occ_diverse_list_dict[token].add(taxon)
                    tokens_occ_dict[token] += 1

    counts_df = pd.DataFrame.from_dict([tokens_occ_dict]).T
    counts_df.columns = ["num_genera_with_token"]

else:
    tokens_occ_dict = defaultdict(int)
    with open(tokenized_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            tokens = line.split(",")[0].split()
            tokens = list(set(tokens))
            for token in tokens:
                tokens_occ_dict[token] += 1
    counts_df = pd.DataFrame.from_dict([tokens_occ_dict]).T
    counts_df.columns = ["num_sequences_with_token"]

tokens_list = [
    key for key, val in tokens_occ_dict.items() if val >= vocab_cutoff_threshold
]
print("num tokens in vocab:", len(tokens_list))
counts_df.to_csv(vocab_counts_path)


# Create vocab.json file

# tokens_list = list(tokens_dict.keys())

special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
all_tokens_list = special_tokens + tokens_list

tokens_dict = {all_tokens_list[i]: i for i in range(len(all_tokens_list))}
json_file = open(vocab_path, "w")
json.dump(tokens_dict, json_file)
json_file.close()
