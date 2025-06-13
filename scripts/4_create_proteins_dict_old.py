import pandas as pd
import pickle
import os
import tqdm
from src.utils import get_project_root
import datetime
import gc
from concurrent.futures import ProcessPoolExecutor
import json
import pyhmmer
import sys

print(sys.argv[1])
with open(sys.argv[1]) as f:
    args_dict = json.load(f)


# This file exists to take the results of the hmm search of the proteins and put them into dictionary form for easily create the gff tokenized sequences
# Lots of consideration to memory usage needed as these dictionaries can be huge
# Im sure there is a better way to do this but this has worked.

# SET THESE EACH RUN AS NEEDED
download_date = args_dict["download_date"]
assembly_source = args_dict[
    "assembly_source"
]  # Where we will get the data from, can be RefSeq or GenBank
num_protein_files = args_dict[
    "num_protein_files"
]  # We split the proteins to be searched for domains into separate files to ease memory issues

filter_by_domain_score = args_dict["filter_by_domain_score"]
if filter_by_domain_score == "True":
    domain_score_filter = args_dict["domain_score_filter"]
else:
    domain_score_filter = None
filter_by_e_value = args_dict["filter_by_e_value"]
if filter_by_e_value == "True":
    e_value_filter = args_dict["e_value_filter"]
else:
    e_value_filter = None
filter_by_gathering_threshold = args_dict["filter_by_gathering_threshold"]

run_name = f"{assembly_source}_{download_date}"
download_folder = os.path.join(get_project_root(), "data", run_name)
proteins_folder = os.path.join(download_folder, "unique_proteins")

pfam_hmm = os.path.join(
    get_project_root(), "data", "Pfam-A.hmm"
)  # Need to have Pfam hmms downloaded and placed here
with pyhmmer.plan7.HMMFile(pfam_hmm) as hmm_file:
    hmms = list(hmm_file)
cutoffs_dict = {}
for hmm in hmms:
    cutoffs_dict[hmm.name.decode("utf-8")] = (
        hmm.cutoffs.gathering2
    )  # We only annotate domains with scores above Pfam provided gathering threshold


start_time = datetime.datetime.now()
print(start_time)


# This is all the columns of data outputted by HMMER, we only need a few
if e_value_filter is None:
    col_names = [
        "target_name",  # name of protein
        # "targ_accession",
        # "tlen",
        "query_name",  # name of Pfam domain
        # "accession",
        # "qlen",
        # "full_seq_e_value",
        # "full_seq_score",
        # "full_seq_bias",
        # "domain_num",
        # "domain_num_of",
        # "domain_c_evalue",
        # "domain_i_evalue",
        "domain_score",
        # "domain_bias",
        # "hmm_from",
        # "hmm_to",
        # "ali_from",
        # "ali_to",
        "env_from",  # roughly where the domain found to begin in the protein
        "env_to",
        # "acc",
        # "description"
    ]
else:
    col_names = [
        "target_name",
        "query_name",
        "domain_i_evalue",
        "domain_score",
        "env_from",
        "env_to",
    ]

domtbl_files = []
for i in range(num_protein_files):
    f = os.path.join(proteins_folder, f"proteins_{i}.domtbl")
    domtbl_files.append(f)


df = pd.DataFrame(columns=col_names)
df["domain_score"] = df["domain_score"].astype(float)
df["env_from"] = df["env_from"].astype(int)
df["env_to"] = df["env_to"].astype(int)
if "domain_i_evalue" in col_names:
    df["domain_i_evalue"] = df["domain_i_evalue"].astype(float)


### ChatGPT helped make the below lines faster
def process_file(hits_file):
    hits = pd.read_csv(
        hits_file,
        delim_whitespace=True,
        comment="#",
        header=None,
        usecols=(
            [0, 3, 13, 19, 20]
            if filter_by_e_value == "False"
            else [0, 3, 12, 13, 19, 20]
        ),
        names=col_names,
    )
    hits["domain_score"] = hits["domain_score"].astype(float)

    if e_value_filter is not None:
        hits["domain_i_evalue"] = hits["domain_i_evalue"].astype(float)
        return hits[
            hits.apply(lambda x: x["domain_i_evalue"] >= e_value_filter, axis=1)
        ]
    elif domain_score_filter is not None:
        return hits[
            hits.apply(lambda x: x["domain_score"] >= domain_score_filter, axis=1)
        ]
    elif filter_by_gathering_threshold == "True":
        gathering_threshold_series = hits["query_name"].map(cutoffs_dict)
        hits = hits[hits["domain_score"] >= gathering_threshold_series]
        return hits
    else:
        return hits


# Filter hits files and create df of domains hits
with ProcessPoolExecutor() as executor:
    hits_frames = list(
        tqdm.tqdm(executor.map(process_file, domtbl_files), total=len(domtbl_files))
    )
df = pd.concat(hits_frames, ignore_index=True)

# Need to save memory
del hits_frames
gc.collect()

print("df created")

df["domain_score"] = df["domain_score"].astype(float)
df["env_from"] = df["env_from"].astype(int)
df["env_to"] = df["env_to"].astype(int)


# Need to sort for later adding the domains in correct order during gff tokenization
df.set_index("target_name", inplace=True)
df.sort_values(["target_name", "domain_score"], ascending=[True, False], inplace=True)

print("df sorted")

# Changing the protein and query names to integers saves memory
# we save the actual names in separate accessions_dict and query_dict dictionaries so we can map them back later.
# Yes i know this code is bad I apologize
accessions_list = list(set(df.index))
accession_dict = {accessions_list[i]: i for i in range(len(accessions_list))}
del accessions_list
df.index = df.index.map(accession_dict)

query_list = list(set(df["query_name"].values))
query_dict = {query_list[i]: i for i in range(len(query_list))}
del query_list
df["query_name"] = df["query_name"].map(query_dict)

print("accession and query converted to int")

with open(os.path.join(proteins_folder, "accessions_dictionary.pkl"), "wb") as f:
    pickle.dump(accession_dict, f)
with open(os.path.join(proteins_folder, "query_dictionary.pkl"), "wb") as f:
    pickle.dump(query_dict, f)

del accession_dict
del query_dict

print("dicts dumped")

# Combining the hits for each protein to a single row, with information separated in each column by spaces. Saves memory
df = df.groupby(df.index).agg(lambda x: " ".join([str(y) for y in x.values.tolist()]))

print("df grouped")

df = df.to_dict("index")

# Combining all columns as strings, saves memory.
for key, value in tqdm.tqdm(df.items()):
    df[key] = str(list(value.values()))

print("df converted to dict")

with open(os.path.join(proteins_folder, "str_proteins_dict.pkl"), "wb") as f:
    pickle.dump(df, f)
