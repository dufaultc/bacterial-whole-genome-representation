import pandas as pd
import os
from src.utils import get_project_root
from collections import defaultdict
import tqdm
import ast
import taxopy
import numpy as np

# SET THESE EACH RUN AS NEEDED
filter_split_date = "april_27"
download_date = "april_22"
assembly_source = "RefSeq"
# Should there be multiple genomes from same species/genus/family?
# If so, how many?
enforce_unique = True
unique_level = "species"
# How many genomes of same unique level should be included at most?
same_rank_allowed_amount_train = "variable"
same_rank_allowed_amount_val = 10
# Split parameters
val_split_taxon = "genus"  # Split train and test data at what taxonomic rank?
validation_ratio = 0.05
test_ratio = 0.05

subset_diverse_species_size = 100000

run_name = f"{assembly_source}_{download_date}"
genomes_folder = os.path.join(get_project_root(), "data", run_name)
tokenized_folder = os.path.join(genomes_folder, "tokenized")

tokenized_file = os.path.join(tokenized_folder, f"tokenized_{run_name}.txt")
accessions_order_file = os.path.join(
    tokenized_folder, f"accessions_order_{run_name}.txt"
)

accessions_file = open(accessions_order_file, "r")
accessions = accessions_file.read().splitlines()
accessions_file.close()


# Based on this metadata, we will determine which accessions to include
metadata_df = pd.read_csv(
    os.path.join(tokenized_folder, f"metadata_df_{run_name}.csv"), header=0, index_col=0
)

# We use taxopy to label each genome with taxonomy
taxdb = taxopy.TaxDb()
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

# We create a test set which has genomes of three different taxonomic distances to the train set
# Genus (no genomes of same genus in the train set), species, and none
unique_values = metadata_df["genus"].unique()
np.random.shuffle(unique_values)
split_index = round(len(unique_values) * test_ratio)
unique_values_test = unique_values[:split_index]
unique_values_train = unique_values[split_index:]
metadata_df_test = metadata_df[metadata_df["genus"].isin(unique_values_test)]
metadata_df_test["split_level"] = "genus"
metadata_df_train = metadata_df[metadata_df["genus"].isin(unique_values_train)]
for tax in ["species"]:
    unique_values = metadata_df_train[tax].unique()
    np.random.shuffle(unique_values)
    split_index = round(len(unique_values) * test_ratio)
    unique_values_test = unique_values[:split_index]
    unique_values_train = unique_values[split_index:]
    new_df = metadata_df_train[metadata_df_train[tax].isin(unique_values_test)]
    new_df["split_level"] = tax
    metadata_df_test = pd.concat([metadata_df_test, new_df])
    metadata_df_train = metadata_df_train[
        metadata_df_train[tax].isin(unique_values_train)
    ]
new_df = metadata_df_train.sample(frac=test_ratio)
new_df["split_level"] = "none"
metadata_df_test = pd.concat([metadata_df_test, new_df])
metadata_df_train = metadata_df_train.drop(new_df.index)

# We also create a validation set which has a specific taxnomic split level as well
unique_values = metadata_df_train[val_split_taxon].unique()
np.random.shuffle(unique_values)
split_index = round(len(unique_values) * validation_ratio)
unique_values_val = unique_values[:split_index]
unique_values_train = unique_values[split_index:]
metadata_df_val = metadata_df_train[
    metadata_df_train[val_split_taxon].isin(unique_values_val)
]
metadata_df_val = (
    metadata_df_val.sample(frac=1)
    .groupby(unique_level)
    .head(same_rank_allowed_amount_val)
)  # Limit the size of the validation set
metadata_df_train_final_split = metadata_df_train[
    metadata_df_train[val_split_taxon].isin(unique_values_train)
]

metadata_df_train_final_split["species_counts"] = metadata_df_train_final_split.groupby(
    ["species"]
)["assembly_info"].transform("count")
metadata_df_train_final_split["genus_counts"] = metadata_df_train_final_split.groupby(
    ["genus"]
)["assembly_info"].transform("count")
metadata_df_train_final_split["family_counts"] = metadata_df_train_final_split.groupby(
    ["family"]
)["assembly_info"].transform("count")

# We take the assemblies which are either type assemblies, have low ANI and coverage with a type asembly, or are from species with 10 or less representatives in the training split
metadata_df_train_final_split = metadata_df_train_final_split[
    (metadata_df_train_final_split["ani_category"] == "type")
    | (
        (metadata_df_train_final_split["ani"] <= 99.5)
        & (metadata_df_train_final_split["coverage"] <= 95)
    )
    | (metadata_df_train_final_split["species_counts"] <= 10)
]
metadata_df_train_final_split = (
    metadata_df_train_final_split.sample(frac=1)
    .groupby([unique_level, "strain"])
    .head(3)
)  # Limit the number of instances of the same strain appearing multiple times


# This function subsets the training split to a certain number of entries
# Entries are selected to maximize the diversity of species present
# For example if N=80, and we have 100 genomes from species A, 50 from species, B, and 10 from species C,
# after running this function the final set will have (35 from species A, 35 from species B, and 10 from species C)
def subset_diverse_species(df, output_size):
    counts = defaultdict(int)  # So we dont have to initialize
    species_indices = defaultdict(list)
    species = df["species"].values.tolist()
    for i, val in enumerate(species):
        counts[val] += 1
        species_indices[val].append(i)
    selected_counts = {taxon: 0 for taxon in species}

    total = 0
    level = 1
    max_count = max(counts.values())
    while level <= max_count and total < output_size:
        for taxon in species:
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
    for taxon in species:
        indices = species_indices[taxon][: selected_counts[taxon]]
        selected_indices.extend(indices)
    return df.iloc[selected_indices]


metadata_df_train_final_split = subset_diverse_species(
    metadata_df_train_final_split, subset_diverse_species_size
)


# Now we upsample genomes from poorly represented species
metadata_df_train_final_split["species_counts"] = metadata_df_train_final_split.groupby(
    ["species"]
)["assembly_info"].transform("count")
metadata_df_train_final_split["upsample_multiple"] = (
    10 // metadata_df_train_final_split["species_counts"]
)
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["upsample_multiple"] == 0, "upsample_multiple"
] = 1
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["upsample_multiple"] == 10, "upsample_multiple"
] = 5
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["species_counts"] == 1, "upsample_multiple"
] = 5
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["species_counts"] == 2, "upsample_multiple"
] = 5
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["species_counts"] == 3, "upsample_multiple"
] = 4
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["species_counts"] == 4, "upsample_multiple"
] = 4
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["species_counts"] == 5, "upsample_multiple"
] = 3
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["species_counts"] == 6, "upsample_multiple"
] = 3
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["species_counts"] == 7, "upsample_multiple"
] = 2
metadata_df_train_final_split.loc[
    metadata_df_train_final_split["species_counts"] == 8, "upsample_multiple"
] = 2
metadata_df_train_final_split = metadata_df_train_final_split.loc[
    metadata_df_train_final_split.index.repeat(
        metadata_df_train_final_split["upsample_multiple"]
    )
].reset_index(drop=True)
metadata_df_train_final_split.set_index("current_accession", inplace=True)
print(metadata_df_train_final_split["species"].value_counts()[:20])

print(metadata_df_train_final_split.shape)
print(metadata_df_val.shape)
print(metadata_df_test.shape)


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
        tokenized_folder,
        f"{filter_split_date}_train_{same_rank_allowed_amount_train}_{unique_level}_split_{val_split_taxon}.csv",
    ),
    "w",
)
train_data_file.write(",".join(header_features) + "\n")
val_data_file = open(
    os.path.join(
        tokenized_folder,
        f"{filter_split_date}_val_{same_rank_allowed_amount_val}_{unique_level}_split_{val_split_taxon}.csv",
    ),
    "w",
)
val_data_file.write(",".join(header_features) + "\n")

test_data_file = open(
    os.path.join(
        tokenized_folder,
        f"{filter_split_date}_test.csv",
    ),
    "w",
)
test_data_file.write(",".join(header_features + ["split_level"]) + "\n")

with open(tokenized_file) as f:
    for i, line in enumerate(f):
        accession = accessions[i]
        if accession in metadata_df_train_final_split.index:
            acc = metadata_df_train_final_split.loc[accession]
            if len(acc.shape) == 1:
                acc = metadata_df_train_final_split.loc[[accession]]
        elif (accession in metadata_df_test.index) or (
            accession in metadata_df_val.index
        ):
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
        if accession in metadata_df_val.index:
            for new_line in new_lines:
                val_data_file.write(new_line + "\n")
        elif accession in metadata_df_test.index:
            for new_line in new_lines:
                test_data_file.write(
                    new_line
                    + f",{str(metadata_df_test.loc[accession]['split_level'])}"
                    + "\n"
                )
        elif accession in metadata_df_train_final_split.index:
            for new_line in new_lines:
                train_data_file.write(new_line + "\n")
val_data_file.close()
train_data_file.close()
test_data_file.close()
