import pandas as pd
import os
from src.utils import get_project_root
from collections import defaultdict
import tqdm
import ast
import taxopy
import numpy as np
import json
import sys


print(sys.argv[1])
with open(sys.argv[1]) as f:
    args_dict = json.load(f)

# SET THESE EACH RUN AS NEEDED
download_run_name = args_dict["download_run_name"]
dataset_name = args_dict["dataset_name"]
validation_split_level = args_dict.get("validation_split_level", "genus")
validation_split_ratio = args_dict.get("validation_split_ratio", 0.05)
test_split_levels = args_dict.get("test_split_levels", ["genus", "species", "none"])
test_split_ratios = args_dict.get("test_split_ratios", [0.05, 0.05, 0.05])
unique_level = args_dict.get("unique_level", "species")
same_rank_allowed_amount_val = args_dict.get("same_rank_allowed_amount_val", None)
do_ani_data_filtering = args_dict.get("do_ani_data_filtering", None)
ani_filter_keep_type_assemblies = args_dict.get("ani_filter_keep_type_assemblies", None)
ani_filter_keep_if_less_than_x_examples = args_dict.get(
    "ani_filter_keep_if_less_than_x_examples", None
)
ani_filter_max = args_dict.get("ani_filter_max", None)
ani_filter_coverage_max = args_dict.get("ani_filter_coverage_max", None)
max_same_strain = args_dict.get("max_same_strain", None)
subset_diverse_to_size = args_dict.get("subset_diverse_to_size", None)
do_upsample = args_dict.get("do_upsample", None)
upsample_multiples = args_dict.get("upsample_multiples", None)


genomes_folder = os.path.join(get_project_root(), "data", download_run_name)
dataset_folder = os.path.join(
    get_project_root(), "data", download_run_name, dataset_name
)
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
tokenized_folder = os.path.join(genomes_folder, "tokenized")

tokenized_file = os.path.join(tokenized_folder, f"tokenized_{download_run_name}.txt")
accessions_order_file = os.path.join(
    tokenized_folder, f"accessions_order_{download_run_name}.txt"
)

accessions_file = open(accessions_order_file, "r")
accessions = accessions_file.read().splitlines()
accessions_file.close()


# Based on this metadata, we will determine which accessions to include
metadata_df = pd.read_csv(
    os.path.join(tokenized_folder, f"metadata_df_{download_run_name}.csv"),
    header=0,
    index_col=0,
)

# We use taxopy to label each genome with taxonomy
taxdb = taxopy.TaxDb(taxdb_dir=genomes_folder)
taxonomies = [
    taxopy.Taxon(
        ast.literal_eval(metadata_df.loc[i]["organism"])["tax_id"],
        taxdb,
    ).rank_name_dictionary
    for i in list(metadata_df.index)
]

metadata_df["species"] = [taxonomies[i].get("species") for i in range(len(taxonomies))]
metadata_df["genus"] = [taxonomies[i].get("genus") for i in range(len(taxonomies))]
metadata_df["family"] = [taxonomies[i].get("family") for i in range(len(taxonomies))]
metadata_df["order"] = [taxonomies[i].get("order") for i in range(len(taxonomies))]
metadata_df["class"] = [taxonomies[i].get("class") for i in range(len(taxonomies))]
metadata_df["phylum"] = [taxonomies[i].get("phylum") for i in range(len(taxonomies))]


if do_ani_data_filtering == "True":
    ani_list = []  # The average nucleotides identity to a type sequence
    ani_category_list = []  # Whether or not the genome is itself a type sequence
    coverage_list = []  # Extimated coverage of genome by a type sequence
    for i in tqdm.tqdm(list(metadata_df.index)):
        try:
            ani_list.append(
                ast.literal_eval(metadata_df.loc[i]["average_nucleotide_identity"])[
                    "best_ani_match"
                ]["ani"]
            )
        except:
            ani_list.append(None)
        try:
            coverage_list.append(
                ast.literal_eval(metadata_df.loc[i]["average_nucleotide_identity"])[
                    "best_ani_match"
                ]["assembly_coverage"]
            )
        except:
            coverage_list.append(None)
        try:
            ani_category_list.append(
                ast.literal_eval(metadata_df.loc[i]["average_nucleotide_identity"])[
                    "category"
                ]
            )
        except:
            ani_category_list.append(None)
    metadata_df["ani"] = ani_list
    metadata_df["ani_category"] = ani_category_list
    metadata_df["coverage"] = coverage_list


if max_same_strain is not None:
    strain_list = []  # Getting strain information from metadata
    for i in tqdm.tqdm(list(metadata_df.index)):
        try:
            strain_list.append(
                ast.literal_eval(metadata_df.loc[i]["organism"])["infraspecific_names"][
                    "strain"
                ]
            )
        except:
            strain_list.append("None")
    metadata_df["strain"] = strain_list


# For each clade, we want to count how common it is in the dataset and add this info to the df
metadata_df["species_counts"] = metadata_df.groupby(["species"])[
    "assembly_info"
].transform("count")
metadata_df["genus_counts"] = metadata_df.groupby(["genus"])["assembly_info"].transform(
    "count"
)
metadata_df["family_counts"] = metadata_df.groupby(["family"])[
    "assembly_info"
].transform("count")
metadata_df["order_counts"] = metadata_df.groupby(["order"])["assembly_info"].transform(
    "count"
)

train_df = metadata_df
test_df = None


def split_by_taxon(train_df, level, split_ratio):
    if level == "none":
        new_test_df = train_df.sample(frac=split_ratio)
        new_test_df["split_level"] = "none"
        train_df = train_df.drop(new_test_df.index)
    else:
        unique_values = train_df[level].unique()
        np.random.shuffle(unique_values)
        split_index = round(len(unique_values) * split_ratio)
        unique_values_test = unique_values[:split_index]
        unique_values_train = unique_values[split_index:]
        new_test_df = train_df[train_df[level].isin(unique_values_test)]
        new_test_df["split_level"] = level
        train_df = train_df.drop(new_test_df.index)
    return new_test_df, train_df


for index, level in enumerate(test_split_levels):
    small_df, train_df = split_by_taxon(train_df, level, test_split_ratios[index])
    if test_df is None:
        test_df = small_df
    else:
        test_df = pd.concat([test_df, small_df])

# We also create a validation set which has a specific taxonomic split level as well
validation_df, train_df = split_by_taxon(
    train_df, validation_split_level, validation_split_ratio
)

if same_rank_allowed_amount_val is not None:
    validation_df = (
        validation_df.sample(frac=1)
        .groupby(unique_level)
        .head(same_rank_allowed_amount_val)
    )  # Limit the size of the validation set

train_df["species_counts"] = train_df.groupby(["species"])["assembly_info"].transform(
    "count"
)
train_df["genus_counts"] = train_df.groupby(["genus"])["assembly_info"].transform(
    "count"
)
train_df["family_counts"] = train_df.groupby(["family"])["assembly_info"].transform(
    "count"
)
print(train_df.shape)
# We take the assemblies which are either type assemblies, have low ANI and coverage with a type asembly, or are from species with 10 or less representatives in the training split
if do_ani_data_filtering == "True":
    keep = train_df["species"] != None
    if ani_filter_max is not None:
        keep = keep & (train_df["ani"] <= ani_filter_max)
    if ani_filter_coverage_max is not None:
        keep = keep & (train_df["coverage"] <= ani_filter_coverage_max)
    if ani_filter_keep_type_assemblies == "True":
        keep = keep | (train_df["ani_category"] == "type")
    if ani_filter_keep_if_less_than_x_examples is not None:
        keep = keep | (
            train_df[f"{unique_level}_counts"]
            <= ani_filter_keep_if_less_than_x_examples
        )
    train_df = train_df[keep]
print(train_df.shape)
if max_same_strain is not None:
    train_df = (
        train_df.sample(frac=1).groupby(["species", "strain"]).head(max_same_strain)
    )  # Limit the number of instances of the same strain appearing multiple times

print(train_df.shape)


# This function subsets the training split to a certain number of entries
# Entries are selected to maximize the diversity of species present
# For example if N=80, and we have 100 genomes from species A, 50 from species, B, and 10 from species C,
# after running this function the final set will have (35 from species A, 35 from species B, and 10 from species C)
def subset_diverse_taxa(df, output_size):
    counts = defaultdict(int)
    taxa_indices = defaultdict(list)
    taxa = df[unique_level].values.tolist()
    for i, val in enumerate(taxa):
        counts[val] += 1
        taxa_indices[val].append(i)
    selected_counts = {taxon: 0 for taxon in list(counts.keys())}

    total = 0
    level = 1
    max_count = max(counts.values())
    while level <= max_count and total < output_size:
        for taxon in list(counts.keys()):
            if (
                selected_counts[taxon] < counts[taxon]
                and selected_counts[taxon] < level
            ):
                selected_counts[taxon] += 1
                total += 1
                if total >= output_size:
                    break
        level += 1
    selected_indices = []
    for taxon in list(counts.keys()):
        indices = taxa_indices[taxon][: selected_counts[taxon]]
        selected_indices.extend(indices)
    return df.iloc[selected_indices]


if subset_diverse_to_size is not None:
    train_df = subset_diverse_taxa(train_df, subset_diverse_to_size)
print("after_subset", train_df.shape)
if do_upsample == "True":
    train_df[f"{unique_level}_counts"] = train_df.groupby([unique_level])[
        "assembly_info"
    ].transform("count")
    if upsample_multiples is None:
        upsample_multiples = [5, 5, 4, 4, 3, 3, 2, 2]
    train_df["upsample_multiple"] = 1
    for i, multiple in enumerate(upsample_multiples):
        train_df.loc[
            train_df[f"{unique_level}_counts"] == (i + 1), "upsample_multiple"
        ] = multiple
    train_df = train_df.loc[
        train_df.index.repeat(train_df["upsample_multiple"])
    ].reset_index(drop=True)
train_df.set_index("current_accession", inplace=True)
print("after_upsample", train_df.shape)
# Writing this stuff to files
header_features = [
    "sequence",
    "accession",
    "species",
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "assembly_level",
    "non_coding_genes",
    "protein_coding_genes",
    "pseudogenes",
    "total_genes",
    "gc_percent",
    "contig_l50",
    "contig_n50",
    "total_sequence_length",
    "total_ungapped_length",
    "number_of_component_sequences",
]


train_data_file = open(
    os.path.join(
        dataset_folder,
        f"train.csv",
    ),
    "w",
)
train_data_file.write(",".join(header_features) + "\n")
val_data_file = open(
    os.path.join(
        dataset_folder,
        f"validation.csv",
    ),
    "w",
)
val_data_file.write(",".join(header_features) + "\n")

test_data_file = open(
    os.path.join(
        dataset_folder,
        f"test.csv",
    ),
    "w",
)
test_data_file.write(",".join(header_features + ["split_level"]) + "\n")

with open(tokenized_file) as f:
    for i, line in enumerate(f):
        accession = accessions[i]
        if accession in train_df.index:
            acc = train_df.loc[accession]
            if len(acc.shape) == 1:
                acc = train_df.loc[[accession]]
        elif (accession in test_df.index) or (accession in validation_df.index):
            acc = metadata_df.loc[[accession]]
        else:
            continue
        new_lines = []
        for j in range(acc.shape[0]):
            new_lines.append(
                ",".join(
                    [
                        line.rstrip().replace(",", ""),
                        accession,
                        str(acc.iloc[j]["species"]).replace(",", ""),
                        str(acc.iloc[j]["genus"]).replace(",", ""),
                        str(acc.iloc[j]["family"]).replace(",", ""),
                        str(acc.iloc[j]["order"]).replace(",", ""),
                        str(acc.iloc[j]["class"]).replace(",", ""),
                        str(acc.iloc[j]["phylum"]).replace(",", ""),
                        ast.literal_eval(acc.iloc[j]["assembly_info"])[
                            "assembly_level"
                        ],
                        str(
                            ast.literal_eval(acc.iloc[j]["annotation_info"])["stats"][
                                "gene_counts"
                            ].get("non_coding")
                        ).replace(",", ""),
                        str(
                            ast.literal_eval(acc.iloc[j]["annotation_info"])["stats"][
                                "gene_counts"
                            ].get("protein_coding")
                        ).replace(",", ""),
                        str(
                            ast.literal_eval(acc.iloc[j]["annotation_info"])["stats"][
                                "gene_counts"
                            ].get("pseudogene")
                        ).replace(",", ""),
                        str(
                            ast.literal_eval(acc.iloc[j]["annotation_info"])["stats"][
                                "gene_counts"
                            ].get("total")
                        ).replace(",", ""),
                        str(
                            ast.literal_eval(acc.iloc[j]["assembly_stats"]).get(
                                "gc_percent"
                            )
                        ).replace(",", ""),
                        str(
                            ast.literal_eval(acc.iloc[j]["assembly_stats"])[
                                "contig_l50"
                            ]
                        ).replace(",", ""),
                        str(
                            ast.literal_eval(acc.iloc[j]["assembly_stats"])[
                                "contig_n50"
                            ]
                        ).replace(",", ""),
                        str(
                            ast.literal_eval(acc.iloc[j]["assembly_stats"])[
                                "total_sequence_length"
                            ]
                        ).replace(",", ""),
                        str(
                            ast.literal_eval(acc.iloc[j]["assembly_stats"])[
                                "total_ungapped_length"
                            ]
                        ).replace(",", ""),
                        str(
                            ast.literal_eval(acc.iloc[j]["assembly_stats"])[
                                "number_of_component_sequences"
                            ]
                        ).replace(",", ""),
                    ]
                )
            )
        if accession in validation_df.index:
            for new_line in new_lines:
                val_data_file.write(new_line + "\n")
        elif accession in test_df.index:
            for new_line in new_lines:
                test_data_file.write(
                    new_line + f",{str(test_df.loc[accession]['split_level'])}" + "\n"
                )
        elif accession in train_df.index:
            for new_line in new_lines:
                train_data_file.write(new_line + "\n")
val_data_file.close()
train_data_file.close()
test_data_file.close()
