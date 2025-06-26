# bacterial-whole-genome-representation

Learn representations of whole bacterial genomes!

Work published in Learning Meaningful Representations of Life Workshop at ICLR 2025 in Singapore. https://www.lmrl.org/

Paper: https://www.biorxiv.org/content/10.1101/2025.04.01.646674v1

This repo is a work in progress, expect more/better/cleaner code to be added in coming weeks.

Current main things missing are:
* Downloadable pretrained models
* Code for running KNN evluations and instructions for getting data from BacDive


## Creating environment

The below commands were run to set up

* `conda create -n "bacterial_env" python=3.12.2` activate
* `pip install -e .`
* `conda install -c conda-forge ncbi-datasets-cli`
* `conda install conda-forge::biopython`
* `conda install anaconda::pandas`
* `conda install conda-forge::tqdm`
* `pip install pyhmmer=0.11.0`
* `conda install bioconda::taxopy`
* `conda install anaconda::scikit-learn`
* `conda install pytorch  pytorch-cuda=12.1 -c pytorch -c nvidia`
* `conda install conda-forge::transformers`
* `pip install hf-mtask-trainer`
* `conda install conda-forge::wandb`
* `pip install mamba-ssm==2.2.2`
* `pip install causal-conv1d==1.4.0`

## Building dataset
* `python ./scripts/1_download_data.py`
* `python ./scripts/2_split_proteins_for_hmmer.py`
* Download `Pfam-A.hmm` from `https://www.ebi.ac.uk/interpro/download/Pfam/`. Unzip and put in data folder.
* `python ./scripts/3_run_hmmer.py`
* `python ./scripts/4_create_proteins_dict.py`
* `python ./scripts/5_run_gff_tokenization.py`
* `python ./scripts/6_filter_split_dataset.py`
* `python ./scripts/7_find_vocab.py`

## Running training

## KNN Evaluation
