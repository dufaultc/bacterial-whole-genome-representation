import pyhmmer
import pandas as pd
from src.utils import get_project_root
import os
import concurrent.futures
import datetime
import itertools
import json
import sys
import subprocess

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
num_hmmer_processes = args_dict["num_hmmer_processes"]
num_cores_per_hmmer_process = args_dict["num_cores_per_hmmer_process"]
block_size_hmmer = args_dict["block_size_hmmer"]
shrink_hmmer_output = args_dict["shrink_hmmer_output"]

pfam_hmm = os.path.join(
    get_project_root(), "data", "Pfam-A.hmm"
)  # Need to have Pfam hmms downloaded and placed here
download_folder = os.path.join(get_project_root(), "data", run_name)
proteins_folder = os.path.join(download_folder, "unique_proteins")

# This can take a really long time!
start_time = datetime.datetime.now()
print(start_time)


# Search all unique proteins for Pfam domains
# We have a separate process for each protein file created earlier
def task_process(task_id):
    print(f"Starting task {task_id}")
    with pyhmmer.plan7.HMMFile(pfam_hmm) as hmm_file:
        hmms = list(hmm_file)

    out_domtbl = os.path.join(
        proteins_folder,
        f"proteins_{task_id}.domtbl",
    )
    print(len(hmms))
    if os.path.isfile(out_domtbl):
        raise FileExistsError("You need to remove this file first")

    reader = open(os.path.join(proteins_folder, f"proteins_{task_id}.faa"), "rb")
    with pyhmmer.easel.SequenceFile(
        reader, digital=True, alphabet=pyhmmer.easel.Alphabet.amino()
    ) as seq_file:
        for batch_id in itertools.count():
            block = seq_file.read_block(
                sequences=block_size_hmmer
            )  # Search 1 million proteins at a time
            if not block:
                break
            else:
                x = 0
                for hits in pyhmmer.hmmsearch(
                    hmms,
                    block,
                    cpus=num_cores_per_hmmer_process,
                ):
                    x = x + 1
                    f = open(out_domtbl, "ab")
                    if x % 1000 == 0:
                        print(batch_id, task_id, x, datetime.datetime.now())
                    hits.write(
                        f, format="domains", header=False
                    )  # write top hmmer hits to domtbl output file
                    f.close()

    reader.close()
    print(f"Task {task_id} completed")
    return f"Task {task_id} completed"


with concurrent.futures.ProcessPoolExecutor(
    max_workers=num_hmmer_processes
) as executor:
    results = list(executor.map(task_process, range(0, num_protein_files)))
# task_process(0)

if shrink_hmmer_output == "True":
    for i in range(0, num_protein_files):
        print(f"Shrinking output file #{i}")
        in_domtbl = os.path.join(
            proteins_folder,
            f"proteins_{i}.domtbl",
        )
        out_domtbl = os.path.join(
            proteins_folder,
            f"proteins_{i}_small.domtbl",
        )
        cmd = f"cat {in_domtbl} | tr -s ' ' | cut -d ' ' -f 1,4,13,14,20,21 > {out_domtbl}"
        result = subprocess.run(cmd, shell=True)


end_time = datetime.datetime.now()
print(start_time)
print(end_time)
