import json
import pandas as pd
from src.utils import get_project_root
import os
import subprocess
import sys

print(sys.argv[1])
with open(sys.argv[1]) as f:
    args_dict = json.load(f)

# SET THESE EACH RUN AS NEEDED
download_date = args_dict["download_date"]
assembly_level = args_dict["assembly_level"]
assembly_source = args_dict[
    "assembly_source"
]  # Where we will get the data from, can be RefSeq or GenBank

no_limit = args_dict["no_limit"]
if no_limit == "False":
    limit = args_dict["limit"]  # The number of available genomes to download
else:
    limit = None

run_name = f"{assembly_source}_{download_date}"
download_folder = os.path.join(
    get_project_root(), "data", run_name
)  # The folder where our data will go
metadata_file = os.path.join(
    download_folder, f"{run_name}_metadata.json"
)  # For each genome we download, NCBI has some associated metadata like genome quality we will use to build our training dataset later
accession_list_file = os.path.join(download_folder, f"{run_name}_accession_list.csv")

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Download metadata for assemblies in source databse
# 2 is bacteria taxid
cmd = f"datasets summary genome taxon 2 \
--annotated \
--report 'genome' \
--exclude-atypical \
--assembly-source {assembly_source} \
--assembly-level {assembly_level} \
--limit {limit if limit is not None else 'all'} \
> {metadata_file}"
output_dir = os.path.join(get_project_root())
subprocess.call(cmd, cwd=output_dir, shell=True)

# Read in the metadata and get the accession numbers, save to accession_list_file
f = open(os.path.join(get_project_root(), "data", metadata_file))
data = json.load(f)
df = pd.DataFrame([x["accession"] for x in data["reports"]])
df.set_index(0, inplace=True)
df.to_csv(accession_list_file, header=False)


# Download gff and protein files for each accession
# This involves downloading a reference to the data (dehydrated), unzipping it, and then rehydration where the referenced data is actually downloaded
cmd = f"datasets download genome accession \
--inputfile {accession_list_file} \
--dehydrated \
--include gff3,protein"  # We need the gff file with the genome annotations, and the protein file with all the predicted proteins in each genome
print(cmd)
subprocess.call(cmd, cwd=download_folder, shell=True)

cmd = f"unzip {os.path.join(download_folder, 'ncbi_dataset.zip')}"
print(cmd)
subprocess.call(cmd, cwd=download_folder, shell=True)

cmd = f"datasets rehydrate --gzip \
--directory {download_folder}"
output_dir = os.path.join(get_project_root())
print(cmd)
subprocess.call(cmd, cwd=output_dir, shell=True)
