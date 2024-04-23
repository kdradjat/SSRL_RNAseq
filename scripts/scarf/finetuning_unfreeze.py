import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append("../")

from scarf.loss import NTXent
from scarf.model import SCARF, SCARF_baseline, MLP, MLP_head_simple, LastLayer, SCARF_SubTab, SCARF_4layers

from example.dataset import ExampleDataset
from example.utils import dataset_embeddings, fix_seed, train_epoch

from cancerclassification.data import *
from cancerclassification.NN import *

##############


# Load Data
print('Loading Data...')
#dataset_pretrain = read_process_data_TCGA('../data/TCGA/pretrain_data.parquet', '../data/TCGA/label.parquet')
#dataset = read_process_data_TCGA('../data/TCGA/nopretrain_data.parquet', '../data/TCGA/label.parquet')
dataset_pretrain = read_process_data_TCGA('../data/TCGA/100Best_pretrain_data.parquet.gzip', '../data/TCGA/label.parquet')
dataset = read_process_data_TCGA('../data/TCGA/100Best_nopretrain_data.parquet.gzip', '../data/TCGA/label.parquet')
nb_classes = len(np.unique(dataset[:,0]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrain_ds = ExampleDataset(
    dataset_pretrain[:,1:],
    dataset_pretrain[:,0]
)

# Training
name = "scarf_unfreeze_4layers_100Best"
logger = LogResults(name, ["prop"])
bs = 8
prop_list = np.arange(0.02, 1.0, 0.01)
corruption_rate = 0.5
dropout = 0.1


for i in range(5) :
    for prop in prop_list :
        # logger
        logger.update_hyps([prop])
        
        # Build and Load model
        # build model
        pretrained_model = SCARF_4layers(
            input_dim=pretrain_ds.shape[1],
            emb_dim=256, # 16 by default, augment in the future
            features_low=pretrain_ds.features_low,
            features_high=pretrain_ds.features_high,
            corruption_rate=corruption_rate,
            dropout=dropout
        ).to(device)

        pretrained_model.load_state_dict(torch.load("saved_models/scarf_4layers_100Best.pt"), strict=False)
        # isolate encoder part
        encoder = pretrained_model.encoder

        # split training set depending on prop value
        idx = generate_indices(dataset, prop=prop, val_prop=0.01, test_prop=0.5, rs=42)

        # get embeddings
        #loader = DataLoader(dataset[:,1:], batch_size=bs, shuffle=False)
        #embeddings = dataset_embeddings(encoder, loader, device)

        # split 
        embeddings_dataset = CancerDatasetTCGA(dataset[:,1:], dataset[:,0], device)
        dataloaders = get_dataloaders(embeddings_dataset, idx, [bs, bs, bs])

        # build MLP
        config_nn = {
            "epochs":100,
            "lr_init":1e-4,
            "early_stop":30,
            "optim":optim.Adam,
            "bn":True,
            "dropout_rate":0.0
        }
        
        head = LastLayer(input_dim=256, output_dim=nb_classes).to(device)
        
        # Merge encoder and MLP
        model = nn.Sequential(encoder, head)
        print(model)
        
        train_nn(config_nn, dataloaders, model, weights=None, early_stop=config_nn["early_stop"], log=False, logger=logger, test=True, mlp_only=True)
        logger.next_run()
        logger.show_progression()

    logger.save_csv()
