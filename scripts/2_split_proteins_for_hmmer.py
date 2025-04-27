import pyhmmer
import pandas as pd
from src.utils import get_project_root
import os
import random
import tqdm
import gzip


# SET THESE EACH RUN AS NEEDED
download_date = "april_28"
assembly_source = "RefSeq"
num_protein_files = 2  # We split the proteins to be searched for domains into separate files to ease memory issues

run_name = f"{assembly_source}_{download_date}"
genomes_folder = os.path.join(get_project_root(), "data", run_name)

# Making a folder where our protein files will go, and creating the files
proteins_folder = os.path.join(genomes_folder, "unique_proteins")
if not os.path.exists(proteins_folder):
    os.makedirs(proteins_folder)
files = []
for i in range(num_protein_files):
    f = open(os.path.join(proteins_folder, f"proteins_{i}.faa"), "wb")
    files.append(f)

# Getting the path to each downloaded accessions data, each has a unique folder
folders_list = [
    accession
    for accession in os.listdir(os.path.join(genomes_folder, "ncbi_dataset", "data"))
    if os.path.isdir(os.path.join(genomes_folder, "ncbi_dataset", "data", accession))
]

# NCBI assigns a unique name to each unique proteins.
# For each accession, read in all the proteins. For each unique protein not seen before, add it to a random protein file.
protein_names = set()
for folder in tqdm.tqdm(folders_list):
    proteins_path = os.path.join(
        genomes_folder, "ncbi_dataset", "data", folder, "protein.faa.gz"
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
