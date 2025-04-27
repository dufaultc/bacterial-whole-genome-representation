## Creating environment

* `conda create -n "bacterial_env" python=3.9` activate
* `pip install -e .`
* `conda install -c conda-forge ncbi-datasets-cli`
* `conda install conda-forge::biopython`
* `conda install anaconda::pandas`
* `conda install conda-forge::tqdm`
* `pip install pyhmmer=0.11.0`
* `conda install bioconda::taxopy`

# Building dataset
* `python ./scripts/1_download_data.py`
* `python ./scripts/2_split_proteins_for_hmmer.py`
* Download `Pfam-A.hmm` from `https://www.ebi.ac.uk/interpro/download/Pfam/`. Unzip and put in data folder.
* `python ./scripts/3_run_hmmer.py`
* `python ./scripts/4_create_proteins_dict.py`
* `python ./scripts/5_run_gff_tokenization.py`
* `python ./scripts/6_filter_split_dataset.py`
* `python ./scripts/7_find_vocab_diverse.py`
