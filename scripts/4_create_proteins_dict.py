import pandas as pd
import pickle
import os
import tqdm
from src.utils import get_project_root
import datetime
import gc
from concurrent.futures import ProcessPoolExecutor


# This file exists to take the results of the hmm search of the proteins and put them into dictionary form for easily create the gff tokenized sequences
# Lots of consideration to memory usage needed as these dictionaries can be huge
# Im sure there is a better way to do this but this has worked.

# SET THESE EACH RUN AS NEEDED
download_date = "april_28"
assembly_source = "RefSeq"
num_protein_files = 2
num_processes = 2
num_cores_per = 4
domain_score_filter = (
    10  # Filtering domain hits, we only take those with domain scores higher than 10
)

run_name = f"{assembly_source}_{download_date}"
genomes_folder = os.path.join(get_project_root(), "data", run_name)
proteins_folder = os.path.join(genomes_folder, "unique_proteins")


start_time = datetime.datetime.now()
print(start_time)


# This is all the columns of data outputted by HMMER, we only need a few
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

domtbl_files = []
for i in range(num_protein_files):
    f = os.path.join(proteins_folder, f"proteins_{i}.domtbl")
    domtbl_files.append(f)


df = pd.DataFrame(columns=col_names)
df["domain_score"] = df["domain_score"].astype(float)
df["env_from"] = df["env_from"].astype(int)
df["env_to"] = df["env_to"].astype(int)


### ChatGPT helped make the below lines faster
def process_file(hits_file):
    hits = pd.read_csv(
        hits_file,
        delim_whitespace=True,
        comment="#",
        header=None,
        usecols=[0, 3, 13, 19, 20],
        names=col_names,
    )
    hits["domain_score"] = hits["domain_score"].astype(float)
    return hits[hits["domain_score"] > domain_score_filter]


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
