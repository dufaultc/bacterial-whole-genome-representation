import os
from src.utils import get_project_root
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import transformers
from transformers import TrainingArguments
import torch
import random
from collections import defaultdict
from transformers import AutoConfig, AutoModelForMaskedLM
from src.modeling_bimamba import (
    BiMambaForMaskedLMAndPresence,
    BiMambaConfig,
)
from src.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForPresence,
    DataCollatorForSOP,
)
from hf_mtask_trainer import HfMultiTaskTrainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# cache_dir = "/media/data/cameron/hf_cache"
os.environ["WANDB_PROJECT"] = "bacterial-domain-learning"
os.environ["HF_DATASETS_CACHE"] = "/media/data/cameron/hf_cache"

model_name_or_path = "yairschiff/bimamba-template"

random.seed(1)

train_date = "feb_5_multi_loss_less_punc"
run_date = "oct_16"
download_date = "aug_12"
assembly_source = "RefSeq"
all_assembles = True
num_assemblies = None

count_vocab_cutoff = 20
count_vocab_occ_cutoff = 20

max_length = 40000  # int(16384 * 2)
cut_contig_start = 1
cut_contig_end = 5
cut_contig_length = 1000
hidden_size = 768
num_hidden_layers = 16
mean_pool = True
weight_decay = 0.1
truncate_odds = 0.15
random_truncation_level = True
mlm_loss_share = 1.0
presence_loss_share = 1.0
mlm_probability = 0.15
dropout_level = 0.20
bidirectional_weight_tie = True
simple_head = True
truncated_one_hot = False

num_epochs = 4
lr = 4e-4
gradient_accumulation_steps = 32

model_type = "bimamba_presence"
# model_type = "bimamba_absence"
# model_type = "bimamba"

# Should there be multiple genomes from same species/genus/family?
# If so, how many?
enforce_unique = True
unique_level = "species"
# How many genomes of same unique level should be included at most?
same_rank_allowed_amount_train = "variable"
# Split parameters
smart_split = True
# Split train and test data at what taxonomic rank?
smart_split_level = "genus"

model_name = f"1000_genus_{model_type}_{run_date}_{train_date}_refseq_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}_{hidden_size}_{num_hidden_layers}_{lr}_{num_epochs}"
# model_name = f"{model_type}_{run_date}_{train_date}_refseq_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}_{hidden_size}_{num_hidden_layers}_{lr}_{num_epochs}"


run_name = f"{assembly_source}_{num_assemblies if not all_assembles else 'all'}_{download_date}"
genomes_folder = os.path.join(get_project_root(), "data", run_name)
models_folder = os.path.join(get_project_root(), "models")
tokenized_folder = os.path.join(genomes_folder, "tokenized")

load_saved = False
load_model_name = f"1000_genus_{"bimamba_presence"}_{run_date}_{"dec_29_multi_loss"}_refseq_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}_{hidden_size}_{num_hidden_layers}_{4e-4}_{4}"
load_path = os.path.join(models_folder, load_model_name, "checkpoint-67016")

vocab_path = os.path.join(
    tokenized_folder,
    f"1000_genus_{run_date}_vocab_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}_count_vocab_cutoff_{count_vocab_cutoff}_occ_cutoff_{count_vocab_occ_cutoff}.json",
    # f"{run_date}_vocab_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}_count_vocab_cutoff_{count_vocab_cutoff}_occ_cutoff_{count_vocab_occ_cutoff}.json",
)

train_data_file = os.path.join(
    tokenized_folder,
    f"{run_date}_train_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}_new.csv",
)

test_data_file = os.path.join(
    tokenized_folder,
    f"{run_date}_val_{same_rank_allowed_amount_train}_{unique_level}_split_{smart_split_level}.csv",
)


vocab = WordLevel.read_file(vocab_path)
wordlevel = WordLevel(vocab, unk_token="[UNK]")
tokenizer = Tokenizer(wordlevel)
tokenizer.pre_tokenizer = WhitespaceSplit()
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    mask_token="[MASK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    eos_token="[SEP]",
    unk_token="[UNK]",
    model_input_names=["input_ids"],
)
print(len(tokenizer.vocab.keys()))


config = BiMambaConfig(
    d_model=hidden_size,
    n_layer=num_hidden_layers,
    pad_token_id=-100,
    bos_token_id=1,
    eos_token_id=2,
    mask_token_id=4,
    unk_token_id=0,
    vocab_size=tokenizer.vocab_size,
    pad_vocab_size_multiple=1,
    max_length=max_length,
    cut_contig_start=cut_contig_start,
    cut_contig_end=cut_contig_end,
    cut_contig_length=cut_contig_length,
    mean_pool=mean_pool,
    truncate_odds=truncate_odds,
    mlm_loss_share=mlm_loss_share,
    presence_loss_share=presence_loss_share,
    random_truncation_level=random_truncation_level,
    dropout_level=dropout_level,
    bidirectional_weight_tie=bidirectional_weight_tie,
    simple_head=simple_head,
    mlm_probability=mlm_probability,
    truncated_one_hot=truncated_one_hot,
    gradient_accumulation_steps=gradient_accumulation_steps,
    load_path=load_path if load_saved else None,
    # return_dict=False,
)
model = BiMambaForMaskedLMAndPresence(config=config)
if load_saved:
    model.load_state_dict(
        torch.load(load_path + "/pytorch_model.bin", weights_only=True), strict=False
    )
print(sum(p.numel() for p in model.parameters()))


ds = load_dataset(
    "csv", data_files={"train": train_data_file, "validation": test_data_file}
)

# ds["train"] = ds["train"].select(list(range(10000)))


def tokenize_function(examples):
    return tokenizer(examples["sequence"])


dataset = ds.map(
    tokenize_function,
    batched=False,
    num_proc=32,
    remove_columns=["sequence"],
    batch_size=100,
)


def shuffle_contigs(examples):
    results = {"input_ids": []}
    for i in range(len(examples["input_ids"])):
        sublists_input_ids = []
        current_sublist_input_ids = []

        for j, item in enumerate(examples["input_ids"][i][1:-1]):
            if item == vocab["contig_start"]:
                if current_sublist_input_ids:
                    sublists_input_ids.append(current_sublist_input_ids)
                current_sublist_input_ids = [vocab["contig_start"]]
            else:
                current_sublist_input_ids.append(item)

        if current_sublist_input_ids:
            sublists_input_ids.append(current_sublist_input_ids)

        new_sublists_input_ids = []
        old_sublists_input_ids = sublists_input_ids
        # for i in range(random.randint(1, 5)):
        for i in range(random.randint(cut_contig_start, cut_contig_end)):
            new_sublists_input_ids = []
            for sublist in old_sublists_input_ids:
                if len(sublist) > cut_contig_length:
                    # Find all indices where the token is vocab["protein_end"]
                    protein_end_indices = [
                        index
                        for index, token in enumerate(sublist)
                        if token == vocab["protein_end"]
                    ]
                    # Filter indices that are at least 100 tokens from the start and end
                    valid_indices = [
                        idx
                        for idx in protein_end_indices
                        if 100 <= idx < len(sublist) - 100
                    ]
                    if valid_indices:
                        # Randomly choose a split index from the valid indices
                        split_index = random.choice(valid_indices)
                        # Split the sublist at the chosen index
                        new_sublists_input_ids.append(sublist[: split_index + 1])
                        new_sublists_input_ids.append(
                            [vocab["contig_start"]] + sublist[split_index + 1 :]
                        )
                else:
                    new_sublists_input_ids.append(sublist)
            old_sublists_input_ids = new_sublists_input_ids

        random.shuffle(old_sublists_input_ids)
        results["input_ids"].append(
            [1] + [item for sublist in old_sublists_input_ids for item in sublist] + [2]
        )
    return results


def truncate_randomly(examples):
    results = {"input_ids": []}
    for i in range(len(examples["input_ids"])):
        if random.random() > truncate_odds:
            length = len(examples["input_ids"][i])
            if random_truncation_level:
                # half_length = round(length * random.uniform(0.4, 0.6))
                half_length = round(length * random.uniform(0.4, 0.6))
            else:
                half_length = length // 2
            start_index = random.randint(1, length - half_length - 1)
            results["input_ids"].append(
                [1]
                + examples["input_ids"][i][start_index : start_index + half_length]
                + [2]
            )
        else:
            if len(examples["input_ids"][i]) > max_length:
                results["input_ids"].append(
                    [1] + examples["input_ids"][i][1 : ((max_length - 2) + 1)] + [2]
                )
            else:
                results["input_ids"].append(examples["input_ids"][i])
    return results


dataset["train"] = dataset["train"].shuffle(seed=1)
dataset["validation"] = dataset["validation"].shuffle(seed=1)
ds = dataset
ds["train"] = ds["train"].map(
    shuffle_contigs,
    batched=True,
    batch_size=20,
    num_proc=32,
)
ds["validation"] = ds["validation"].map(
    shuffle_contigs,
    batched=True,
    batch_size=20,
    num_proc=32,
)
print(ds)


def one_hot_function(examples):
    results = {"one_hots": []}
    for i in range(len(examples["input_ids"])):
        one_hot_dict = {j: 0 for j in range(len(vocab.keys()))}
        ids = set(examples["input_ids"][i])
        for key in one_hot_dict.keys():
            if key in ids:
                one_hot_dict[key] = 1
        results["one_hots"].append(list(one_hot_dict.values()))
    return results


ds = ds.map(one_hot_function, batched=True, num_proc=16)
ds["train"] = ds["train"].map(
    truncate_randomly,
    batched=True,
    batch_size=20,
    num_proc=16,
)
ds["validation"] = ds["validation"].map(
    truncate_randomly,
    batched=True,
    batch_size=20,
    num_proc=16,
)
# ds["train"] = ds["train"].select(list(range(100000)))

ds = ds.remove_columns(
    [x for x in ds["train"].features.keys() if x not in ["one_hots", "input_ids"]]
)
print(ds)

training_args = TrainingArguments(
    output_dir=f"{models_folder}/{model_name}",
    run_name=model_name,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    logging_steps=20,
    save_steps=400,
    save_total_limit=5,
    eval_steps=1500,
    push_to_hub=False,
    num_train_epochs=num_epochs,
    report_to="wandb",
    gradient_accumulation_steps=gradient_accumulation_steps,
    dataloader_num_workers=8,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=2,
    bf16=True,
    weight_decay=weight_decay,
    learning_rate=lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.10,
    save_safetensors=False,
    label_names=["labels", "one_hots"],
    adam_beta1=0.9,
    adam_beta2=0.95,
)


print(sum(p.numel() for p in model.parameters()))

data_collator = DataCollatorForPresence(
    tokenizer=tokenizer,
    mlm_probability=mlm_probability,
)


trainer = HfMultiTaskTrainer(
    # trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
    callbacks=[transformers.integrations.WandbCallback],
)


# if load_saved:
#     trainer.train(
#         resume_from_checkpoint=load_path,
#     )
# else:
trainer.train()
