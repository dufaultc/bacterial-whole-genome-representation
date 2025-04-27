import pandas as pd
import pickle
import os
import tqdm
from src.utils import get_project_root
import datetime
import json
from collections import defaultdict


filter_split_date = "april_28"
download_date = "april_28"
assembly_source = "RefSeq"
# Should there be multiple genomes from same species/genus/family?
# If so, how many?
enforce_unique = True
unique_level = "species"
# How many genomes of same unique level should be included at most?
same_rank_allowed_amount_train = "variable"

# Split train and test data at what taxonomic rank?
smart_split_level = "genus"

run_name = f"{assembly_source}_{download_date}"
genomes_folder = os.path.join(get_project_root(), "data", run_name)
tokenized_folder = os.path.join(genomes_folder, "tokenized")
vocab_path = os.path.join(
    tokenized_folder,
    f"1000_genus_{filter_split_date}_vocab_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}.json",
)
vocab_counts_path = os.path.join(
    tokenized_folder,
    f"genus_{filter_split_date}_counts_vocab_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}.csv",
)

start_time = datetime.datetime.now()
print(start_time)

train_data_file = os.path.join(
    tokenized_folder,
    f"{filter_split_date}_train_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}.csv",
)

tokens_occ_genus_dict = defaultdict(int)
tokens_occ_genus_list_dict = defaultdict(set)
with open(train_data_file) as f:
    for line in f:
        tokens = line.split(",")[0].split()
        genus = line.split(",")[3]
        tokens = list(set(tokens))
        for token in tokens:
            if genus not in tokens_occ_genus_list_dict[token]:
                tokens_occ_genus_list_dict[token].add(genus)
                tokens_occ_genus_dict[token] += 1

counts_df = pd.DataFrame.from_dict([tokens_occ_genus_dict]).T
counts_df.columns = ["appears_in_genus"]
counts_df.to_csv(vocab_counts_path)


# Create vocab.json file
tokens_list = [key for key, val in tokens_occ_genus_dict.items() if val >= 1000]
# tokens_list = list(tokens_dict.keys())

special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
all_tokens_list = special_tokens + tokens_list

tokens_dict = {all_tokens_list[i]: i for i in range(len(all_tokens_list))}
json_file = open(vocab_path, "w")
json.dump(tokens_dict, json_file)
json_file.close()
