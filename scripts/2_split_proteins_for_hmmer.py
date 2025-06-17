import pyhmmer
import pandas as pd
from src.utils import get_project_root
import os
import random
import tqdm
import gzip
import json
import sys
import subprocess
import re

print(sys.argv[1])
with open(sys.argv[1]) as f:
    args_dict = json.load(f)


# SET THESE EACH RUN AS NEEDED
run_name = args_dict["run_name"]
assembly_source = args_dict[
    "assembly_source"
]  # Where we will get the data from, can be RefSeq or GenBank
num_protein_files = args_dict[
    "num_protein_files"
]  # We split the proteins to be searched for domains into separate files to ease memory issues

reuse_annotations_from = args_dict.get("reuse_annotations_from", None)

download_folder = os.path.join(get_project_root(), "data", run_name)

# Making a folder where our protein files will go, and creating the files
proteins_folder = os.path.join(download_folder, "unique_proteins")
# if not os.path.exists(proteins_folder):
# os.makedirs(proteins_folder)
files = []
for i in range(num_protein_files):
    f = open(os.path.join(proteins_folder, f"proteins_{i}_test.faa"), "wb")
    files.append(f)

# Getting the path to each downloaded accessions data, each has a unique folder
folders_list = [
    accession
    for accession in os.listdir(os.path.join(download_folder, "ncbi_dataset", "data"))
    if os.path.isdir(os.path.join(download_folder, "ncbi_dataset", "data", accession))
]

if reuse_annotations_from is not None:
    pattern = re.compile(r"^proteins_\d+\.faa$")
    existing_files_path = os.path.join(
        get_project_root(), "data", reuse_annotations_from, "unique_proteins"
    )
    existing_files = os.listdir(existing_files_path)
    existing_protein_files = [
        os.path.join(existing_files_path, file)
        for file in existing_files
        if pattern.match(file)
    ]
    accessions_file_name = os.path.join(
        proteins_folder, f"annotated_in_{reuse_annotations_from}.txt"
    )
    for file in existing_protein_files:
        cmd = f"cat {file} | grep '>' | tr -d '>' | cut -d ' ' -f 1 >> {accessions_file_name}."

    with open(accessions_file_name, "r") as f:
        old_proteins = f.read().splitlines()
    print(len(old_proteins))
    protein_names = set(old_proteins)
else:
    protein_names = set()

# NCBI assigns a unique name to each unique proteins.
# For each accession, read in all the proteins. For each unique protein not seen before, add it to a random protein file.
for folder in tqdm.tqdm(folders_list):
    proteins_path = os.path.join(
        download_folder, "ncbi_dataset", "data", folder, "protein.faa.gz"
    )
    reader = gzip.open(proteins_path, "r")
    for sequence in pyhmmer.easel.SequenceFile(reader, digital=True):
        if sequence.name in protein_names:
            continue
        else:
            sequence.write(files[random.randrange(num_protein_files)])
            protein_names.add(sequence.name)
    reader.close()

for i in range(num_protein_files):
    files[i].close()
