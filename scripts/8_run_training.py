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
from src.modeling_bimamba import (
    BiMambaForMaskedLMAndPresence,
    BiMambaConfig,
)
from src.data_collator import (
    DataCollatorForPresence,
)
from hf_mtask_trainer import HfMultiTaskTrainer
import json
import sys

print(sys.argv[1])
with open(sys.argv[1]) as f:
    args_dict = json.load(f)


model_name = args_dict["model_name"]
vocab_name = args_dict["vocab_name"]
download_run_name = args_dict["download_run_name"]
dataset_name = args_dict["dataset_name"]


dataset_folder = os.path.join(
    get_project_root(), "data", download_run_name, dataset_name
)
model_folder = os.path.join(dataset_folder, model_name)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

os.environ["WANDB_PROJECT"] = args_dict["wandb_project"]
os.environ["HF_DATASETS_CACHE"] = args_dict["hf_datasets_cache"]


random.seed(1)

max_length = args_dict.get("max_length", 40000)
truncate_odds = args_dict.get("truncate_odds", 0.85)
cut_contig_start = args_dict.get("cut_contig_start", 1)
cut_contig_end = args_dict.get("cut_contig_end", 5)
cut_contig_length = args_dict.get("cut_contig_length", 5)


hidden_size = args_dict.get("hidden_size", 768)
num_hidden_layers = args_dict.get("num_hidden_layers", 16)
mlm_loss_multiplier = args_dict.get("mlm_loss_multiplier", 1.0)
presence_loss_multiplier = args_dict.get("presence_loss_multiplier", 1.0)
mlm_probability = args_dict.get("mlm_probability", 0.15)
dropout_level = args_dict.get("dropout_level", 0.2)

weight_decay = args_dict.get("weight_decay", 16)
num_epochs = args_dict.get("num_epochs", 4)
lr = args_dict.get("lr", 4e-4)
gradient_accumulation_steps = args_dict.get("gradient_accumulation_steps", 32)

logging_steps = args_dict.get("logging_steps", 20)
save_steps = args_dict.get("save_steps", 400)
save_total_limit = args_dict.get("save_total_limit", 5)
eval_steps = args_dict.get("eval_steps", 1500)

vocab_path = os.path.join(
    dataset_folder,
    f"{vocab_name}_vocab.json",
)

train_data_file = os.path.join(
    dataset_folder,
    f"train.csv",
)

validation_data_file = os.path.join(
    dataset_folder,
    f"validation.csv",
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
    mean_pool=True,
    truncate_odds=truncate_odds,
    mlm_loss_multiplier=mlm_loss_multiplier,
    presence_loss_multiplier=presence_loss_multiplier,
    random_truncation_level=True,
    dropout_level=dropout_level,
    bidirectional_weight_tie=True,
    simple_head=True,
    mlm_probability=mlm_probability,
    truncated_one_hot=False,
    gradient_accumulation_steps=gradient_accumulation_steps,
)
model = BiMambaForMaskedLMAndPresence(config=config)


ds = load_dataset(
    "csv", data_files={"train": train_data_file, "validation": validation_data_file}
)

dataset = ds.map(
    lambda x: tokenizer(x["sequence"]),
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

        for j, token in enumerate(examples["input_ids"][i][1:-1]):
            if token == vocab["contig_start"]:
                if current_sublist_input_ids:
                    sublists_input_ids.append(current_sublist_input_ids)
                current_sublist_input_ids = [vocab["contig_start"]]
            else:
                current_sublist_input_ids.append(token)

        if current_sublist_input_ids:
            sublists_input_ids.append(current_sublist_input_ids)

        new_sublists_input_ids = []
        old_sublists_input_ids = sublists_input_ids

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
        if random.random() > 1 - truncate_odds:
            length = len(examples["input_ids"][i])
            half_length = round(length * random.uniform(0.4, 0.6))
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

ds = ds.remove_columns(
    [x for x in ds["train"].features.keys() if x not in ["one_hots", "input_ids"]]
)
print(ds)

training_args = TrainingArguments(
    output_dir=model_folder,
    run_name=model_name,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    eval_steps=eval_steps,
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
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
    callbacks=[transformers.integrations.WandbCallback],
)

trainer.train()
