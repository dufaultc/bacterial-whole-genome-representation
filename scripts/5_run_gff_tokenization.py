import pandas as pd
import os
from src.utils import get_project_root
import concurrent.futures
import datetime
from collections import defaultdict
import tqdm
import gzip
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
import ast
import gc

"""
This file goes through genome assemblies downloaded earlier
 and "converts" them to sequences of genomic elements. 
 Requires the results of the protein domain search performed earlier.
 Each genome assembly file is opened and  elements added to the growing sequence
 in the order they appears. For each protein encountered, domains are looked up
 in the proteins_dict created in the previous step. 

 The genomic element sequences consist of:
    - Pfam domains. Tokens are just the domain names. 
    - tRNA. Tokens indicate the type (e.x. tRNA-thr)
    - For other non-coding RNA (rRNA, tmRNA, etc.), if an Rfam families is available it is added, otherwise nothing added.
    - Pseudogenes. Token is just "pseudogene"
    - Named repeats if come across. Mostly just "CRISPR".
    - "Punctuation" tokens. contig_start added when each new contig found, 
    protein_start and protein_end added for each coding gene found.
    - Strand tokens "+" and "-" are added when next genomic element is on opposite strand of the previous one.
"""

# SET THESE EACH RUN AS NEEDED
download_date = "may_13"
assembly_source = "RefSeq"
num_processes = 2


run_name = f"{assembly_source}_{download_date}"
genomes_folder = os.path.join(get_project_root(), "data", run_name)
proteins_folder = os.path.join(genomes_folder, "unique_proteins")
tokenized_folder = os.path.join(genomes_folder, "tokenized")
metadata_file = os.path.join(genomes_folder, f"{run_name}_metadata.json")


if not os.path.exists(tokenized_folder):
    os.makedirs(tokenized_folder)

start_time = datetime.datetime.now()
print(start_time)


folders_list = [
    accession
    for accession in sorted(
        os.listdir(os.path.join(genomes_folder, "ncbi_dataset", "data"))
    )
    if os.path.isdir(os.path.join(genomes_folder, "ncbi_dataset", "data", accession))
]

# Saving the metadata in a separate df
meta_f = open(metadata_file, "r")
data = json.load(meta_f)
df = pd.DataFrame([x for x in data["reports"] if x["accession"] in folders_list])
df.set_index("accession", inplace=True)
df.sort_index(inplace=True)
df.to_csv(os.path.join(tokenized_folder, f"metadata_df_{run_name}.csv"), header=True)

# We created these last step, we use them to map proteins we encounter to domains
with open(os.path.join(proteins_folder, "accessions_dictionary.pkl"), "rb") as f:
    accessions_dictionary = pickle.load(f)
with open(os.path.join(proteins_folder, "query_dictionary.pkl"), "rb") as f:
    query_dictionary = pickle.load(f)
query_dictionary = {v: k for k, v in query_dictionary.items()}

with open(os.path.join(proteins_folder, "str_proteins_dict.pkl"), "rb") as f:
    proteins_dict = pickle.load(f)

print("loaded")
print(datetime.datetime.now())


tokenized_file = os.path.join(tokenized_folder, f"tokenized_{run_name}.txt")
accessions_order_file = os.path.join(
    tokenized_folder, f"accessions_order_{run_name}.txt"
)  # Ass we put the tokenized genomes in tokenized_file, we keep track of the accessions added in this file so we dont lose track of their order


out_tokenized = open(tokenized_file, "w")
out_tokenized.close()
out_accessions_order = open(accessions_order_file, "w")
out_accessions_order.close()


# Tokenizes each accession
def tokenize_gff(accession):
    in_gff = os.path.join(
        genomes_folder, "ncbi_dataset", "data", accession, f"genomic.gff.gz"
    )
    tokens = []
    last_strand = "+"
    ranges = defaultdict(set)
    starts = defaultdict(dict)
    with gzip.open(in_gff, "rt") as gff_f:
        for line in gff_f:
            if line[0] == "#":  # Skip the comment lines
                continue

            features = line.split("\t")  # Information in the gff files is tab separated
            attributes = {
                feature.split("=")[0]: feature.split("=")[1]
                for feature in features[8].rstrip().split(";")
                if "=" in feature
            }  #
            if features[2] == "region":
                tokens.append("contig_start")
                continue
            if features[2] == "gene" or features[2] == "exon":
                continue
            if features[6] != last_strand:
                tokens.append(features[6])
                last_strand = features[6]

            if features[2] == "CDS":
                if "pseudo" in attributes.keys():
                    if attributes["pseudo"] == "true":
                        continue
                else:
                    tokens.append("protein_start")
                    try:
                        doms = ast.literal_eval(
                            proteins_dict[accessions_dictionary[attributes["Name"]]]
                        )
                    except:
                        tokens.append("protein_end")
                        continue
                    for i in range(len(doms[0].split())):
                        r = set(
                            range(
                                int(doms[2].split()[i]),  # env_from
                                int(doms[3].split()[i]),  # env_to
                            )
                        )
                        if not ranges[attributes["Name"]].intersection(r):
                            ranges[attributes["Name"]].update(r)
                            starts[attributes["Name"]][int(doms[2].split()[i])] = (
                                query_dictionary[int(doms[0].split()[i])]
                            )
                    for x in sorted(list(starts[attributes["Name"]].keys())):
                        tokens.append(starts[attributes["Name"]][x])
                    tokens.append("protein_end")
            elif features[2] == "pseudogene":
                tokens.append("pseudogene")
            elif features[2] == "tRNA" or features[2] == "pseudogenic_tRNA":
                tokens.append(attributes["product"])
            elif features[2] == "direct_repeat":
                tokens.append(attributes["rpt_family"])
            elif features[2] in [
                "riboswitch",
                "tmRNA",
                "RNase_P_RNA",
                "SRP_RNA",
                "binding_site",
                "rRNA",
                "ncRNA",
                "antisense_RNA",
                "hammerhead_ribozyme",
            ]:
                try:
                    token = attributes["Dbxref"]
                    if "RFAM" in token:
                        token = token[:12]
                        tokens.append(token)
                    else:
                        print(features[2], features[0], accession)
                except:
                    print(features[2], features[0], accession)
            elif features[2] == "sequence_feature":
                try:
                    tokens.append(attributes["Dbxref"])
                except:
                    tokens.append("misc")
            else:
                print(features[2], features[0], accession)
    return " ".join(tokens), accession


cumsum = 0
for start in tqdm.tqdm(range(0, len(folders_list), 1000)):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(tokenize_gff, folder)
            for folder in tqdm.tqdm(
                folders_list[start : start + 1000],
                miniters=100,
                mininterval=20,
                maxinterval=30,
            )
        ]
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            cumsum += 1
            tokens, accession = future.result()
            out_tokenized = open(tokenized_file, "a")
            out_accessions_order = open(accessions_order_file, "a")
            out_tokenized.write(tokens + "\n")
            out_accessions_order.write(accession + "\n")
            out_tokenized.close()
            out_accessions_order.close()
            del tokens, accession, future
            if cumsum % 1000 == 0:
                gc.collect()

end_time = datetime.datetime.now()
print(start_time)
print(end_time)
