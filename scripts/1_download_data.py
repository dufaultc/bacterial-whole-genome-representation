import json
import pandas as pd
from src.utils import get_project_root
import os
from Bio import Entrez
import subprocess
import sys
import random
import taxopy
import ast

date = "march_12"
assembly_level = "complete,chromosome,scaffold,contig"
files_to_include = "gff3,protein"
assembly_source = "RefSeq"

run_name = f"{assembly_source}_{date}"

genomes_folder = os.path.join(get_project_root(), "data", run_name)
metadata_file = os.path.join(genomes_folder, f"{run_name}_metadata.json")
accession_list_file = os.path.join(genomes_folder, f"{run_name}_accession_list.csv")

if not os.path.exists(genomes_folder):
    os.makedirs(genomes_folder)

# Download metadata for all assemblies in source databse
# 2 is bacteria taxid
# cmd = f"datasets summary genome taxon 2 \
# --annotated \
# --report 'genome' \
# --exclude-atypical \
# --assembly-source {assembly_source} \
# --assembly-level {assembly_level} \
# > {metadata_file}"
# output_dir = os.path.join(get_project_root())
# subprocess.call(cmd, cwd=output_dir, shell=True)


# f = open(os.path.join(get_project_root(), "data", metadata_file))
# data = json.load(f)

# df = pd.DataFrame([x["accession"] for x in data["reports"]])
# df.set_index(0, inplace=True)
# df.to_csv(accession_list_file, header=False)


cmd = f"datasets download genome accession \
--inputfile {accession_list_file} \
--dehydrated \
--include {files_to_include}"

print(cmd)
subprocess.call(cmd, cwd=genomes_folder, shell=True)

cmd = f"unzip {os.path.join(genomes_folder, 'ncbi_dataset.zip')}"
print(cmd)
subprocess.call(cmd, cwd=genomes_folder, shell=True)

cmd = f"datasets rehydrate --gzip \
--directory {genomes_folder}"
output_dir = os.path.join(get_project_root())
print(cmd)
subprocess.call(cmd, cwd=output_dir, shell=True)
