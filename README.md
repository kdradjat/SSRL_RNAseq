# Self-Supervised Learning for Gene Expression Data

This repository contains a few state-of-the-art **self-supervised learning** methods for tabular data, which has been a popular topic recently.<br>
The goal of this repository is to apply these methods to **gene expression data** (TCGA), in order to build good data representations and foundation models.<br>

## Methods and Papers
The methods tested so far are : 
### Generative Approach :
* VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (NeurIPS'20) [[Paper]](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf) [[Supplementary]](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Supplemental.pdf) [[Code]](https://github.com/jsyoon0823/VIME)

### Contrastive Approach :
* SCARF:Self-Supervised Contrastive Learning using Random Feature Corruption (ICLR'22) [[Paper]](https://arxiv.org/pdf/2106.15147.pdf) [[Code]](https://github.com/clabrugere/pytorch-scarf) 
* BYOL: Bootstrap Your Own Latent: A new approach to self-supervised Learning: A new approach to self-supervised Learning (2020) [[Paper]](https://arxiv.org/pdf/2006.07733.pdf) [[Code]](https://github.com/lucidrains/byol-pytorch)

## Dataset
The Cancer Genome Atlas ([[TCGA]](https://portal.gdc.cancer.gov/)) collected many types of data for each of over 20,000 tumor and normal samples. Each step in the Genome Characterization Pipeline generated numerous data points, such as:
* clinical information (e.g., smoking status)
* molecular analyte metadata (e.g., sample portion weight)
* molecular characterization data (e.g., gene expression values) <br>
Pretraining and finetuning datasets can be found [[here]](https://drive.google.com/drive/folders/13wjd7KRhvVeLaCcsVvXKM7Tnr4HGhdZt?usp=sharing).

## Installation
``` console
git clone git@github.com:kdradjat/SSRL_RNAseq.git
cd SSRL_RNAseq
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Usage
All the scripts used for pretraining and finetuning with the differents methods are available on the [scripts](https://github.com/kdradjat/SSRL_RNAseq/tree/main/scripts) folder.

### SCARF
#### Pretraining:
##### Example command
```
python pretraining.py --data_file <pretraining_filepath> --batch_size <batch_size> --epoch <nb_epoch> --model_name <saved_model_name> --history_name <loss_history_file_name> 
```

#### Finetuning:
##### Example command
```
python finetuning_unfreeze.py --pretraining_file <pretraining_data_filepath> --finetuning_file <finetuning_data_filepath> --label_file <finetuning_label_filepath> --batch_size <batch_size> --epoch <nb_epoch> --model_name <used_model_path> --history_name <acc_history_file_name>
```

### VIME
#### Pretraining:
##### Example command
```
python pretraining.py --data_file <pretraining_filepath> --batch_size <batch_size> --epoch <nb_epoch> --model_name <saved_model_name> --history_name <loss_history_file_name>
```
#### Finetuning:
```
python finetuning.py --data_file <finetuning_data_filepath> --label_file <finetuning_label_filepath> --batch_size <batch_size> --epoch <nb_epoch> --model_name <used_model_path> --history_name acc_history_file_name>
```

### BYOL
#### Pretraining:

#### Finetuning:

