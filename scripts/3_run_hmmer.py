import pyhmmer
import pandas as pd
from src.utils import get_project_root
import os
import concurrent.futures
import datetime
import itertools

# SET THESE EACH RUN AS NEEDED
download_date = "april_28"
assembly_source = "RefSeq"
num_protein_files = 2
num_processes = 2
num_cores_per = 4

pfam_hmm = os.path.join(
    get_project_root(), "data", "Pfam-A.hmm"
)  # Need to have Pfam hmms downloaded and placed here
run_name = f"{assembly_source}_{download_date}"
genomes_folder = os.path.join(get_project_root(), "data", run_name)
proteins_folder = os.path.join(genomes_folder, "unique_proteins")

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
    with pyhmmer.easel.SequenceFile(reader, digital=True) as seq_file:
        for batch_id in itertools.count():
            block = seq_file.read_block(
                sequences=1000000
            )  # Search 1 million proteins at a time
            if not block:
                break
            else:
                x = 0
                for hits in pyhmmer.hmmsearch(hmms, block, cpus=num_cores_per):
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


with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    results = list(executor.map(task_process, range(0, num_processes)))

end_time = datetime.datetime.now()
print(start_time)
print(end_time)
