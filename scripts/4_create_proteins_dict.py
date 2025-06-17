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
import re
import math

print(sys.argv[1])
with open(sys.argv[1]) as f:
    args_dict = json.load(f)


# This file exists to take the results of the hmm search of the proteins and put them into dictionary form for easily create the gff tokenized sequences
# Lots of consideration to memory usage needed as these dictionaries can be huge
# Im sure there is a better way to do this but this has worked.

# SET THESE EACH RUN AS NEEDED
num_chunks = 20
run_name = args_dict["run_name"]
assembly_source = args_dict[
    "assembly_source"
]  # Where we will get the data from, can be RefSeq or GenBank
num_protein_files = args_dict[
    "num_protein_files"
]  # We split the proteins to be searched for domains into separate files to ease memory issues

reuse_annotations_from = args_dict.get("reuse_annotations_from", None)
shrink_hmmer_output = args_dict.get("shrink_hmmer_output", "False")

domain_score_filter = args_dict.get("domain_score_filter", None)
e_value_filter = args_dict.get("e_value_filter", None)
filter_by_gathering_threshold = args_dict.get("filter_by_gathering_threshold", "False")

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
        "env_from",  # roughly where the domain found in the protein
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
    f = os.path.join(
        proteins_folder,
        (
            f"proteins_{i}_small.domtbl"
            if shrink_hmmer_output == "True"
            else f"proteins_{i}.domtbl"
        ),
    )
    domtbl_files.append(f)
if reuse_annotations_from is not None:
    if shrink_hmmer_output == "True":
        pattern = re.compile(r"^proteins_\d+\_small.domtbl$")
    else:
        pattern = re.compile(r"^proteins_\d+\.domtbl$")
    existing_files_path = os.path.join(
        get_project_root(), "data", reuse_annotations_from, "unique_proteins"
    )
    existing_annotation_files = [
        os.path.join(existing_files_path, file)
        for file in os.listdir(existing_files_path)
        if pattern.match(file)
    ]
    for file in existing_annotation_files:
        f = os.path.join(existing_files_path, file)
        domtbl_files.append(f)

df = pd.DataFrame(columns=col_names)
df["domain_score"] = df["domain_score"].astype(float)


def process_file(hits_file):
    hits = pd.read_csv(
        hits_file,
        sep=" ",
        header=None,
        usecols=([0, 1, 3, 4, 5] if e_value_filter is None else [0, 1, 2, 3, 4, 5]),
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

# Need to sort for later adding the domains in correct order during gff tokenization
df.set_index("target_name", inplace=True)

accessions_list = list(set(df.index))
accession_dict = {accessions_list[i]: i for i in range(len(accessions_list))}
max_len = len(accessions_list)
del accessions_list
df.index = df.index.map(accession_dict)

query_list = list(set(df["query_name"].values))
query_dict = {query_list[i]: i for i in range(len(query_list))}
del query_list
df["query_name"] = df["query_name"].map(query_dict)


# Changing the protein and query names to integers saves memory
# we save the actual names in separate accessions_dict and query_dict dictionaries so we can map them back later.
print("accession and query converted to int")

with open(os.path.join(proteins_folder, "accessions_dictionary.pkl"), "wb") as f:
    pickle.dump(accession_dict, f)
with open(os.path.join(proteins_folder, "query_dictionary.pkl"), "wb") as f:
    pickle.dump(query_dict, f)

del accession_dict
del query_dict

print("dicts dumped")


def groupby_chunk(chunk_df):
    chunk_df.sort_values(
        ["target_name", "domain_score"], ascending=[True, False], inplace=True
    )
    return chunk_df.groupby(chunk_df.index).agg(
        lambda x: " ".join(
            [str(y) for y in x.values.tolist()]
        )  # Combining the hits for each protein to a single row, with information separated in each column by spaces. Saves memory
    )


step = math.ceil(max_len / num_chunks)
start_ends = [(i, min(i + step, max_len)) for i in range(0, max_len, step)]
chunks = [df[(df.index >= start) & (df.index < end)] for start, end in start_ends]
with ProcessPoolExecutor(max_workers=num_chunks) as executor:
    results = list(executor.map(groupby_chunk, chunks))
df = pd.concat(results)
print("df grouped")

df = df.to_dict("index")

# Combining all columns as strings, saves memory.
for key, value in tqdm.tqdm(df.items()):
    df[key] = str(list(value.values()))

print("df converted to dict")

with open(os.path.join(proteins_folder, "str_proteins_dict.pkl"), "wb") as f:
    pickle.dump(df, f)

end_time = datetime.datetime.now()
print(start_time)
print(end_time)
