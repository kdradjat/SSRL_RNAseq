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
*clinical information (e.g., smoking status)
*molecular analyte metadata (e.g., sample portion weight)
*molecular characterization data (e.g., gene expression values)
