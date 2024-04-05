#%load_ext autoreload
#%autoreload 2

import sys

import os

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
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from scarf.loss import NTXent, BarlowTwins
from scarf.model import SCARF, MLP, SCARF_modified
from cancerclassification.data import *
from cancerclassification.NN import *

from example.dataset import ExampleDataset
from example.utils import dataset_embeddings, fix_seed, train_epoch

#################


# Load data
x_unlabel = read_process_data_TCGA_unlabel('../data/TCGA/pretrain_data.parquet')
#x_unlabel = np.load('balanced_pretrain.npy')
#x_unlabel = x_unlabel[:,1:]
print(x_unlabel.shape)
print(x_unlabel.min(axis=0))
print(x_unlabel.max(axis=0))
train_dataset = read_process_data_TCGA('../data/TCGA/train_data.parquet', '../data/TCGA/label.parquet')

# Training
corruption_rate = 0.3
dropout = 0.1
batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_loader = DataLoader(x_unlabel, batch_size=batch_size, shuffle=True)

# build model
model = SCARF_modified(
    input_dim=x_unlabel.shape[1],
    emb_dim=64, # 16 by default, augment in the future
    features_low=x_unlabel.min(axis=0),
    features_high=x_unlabel.max(axis=0),
    nb_classes=len(np.unique(train_dataset[:,0])),
    corruption_rate=corruption_rate,
    dropout=dropout
).to(device)


import time
epochs = 1000

optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
#optimizer = Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)
ntxent_loss = NTXent()
#ntxent_loss = BarlowTwins()

loss_history = []

start = time.time()
for epoch in range(1, epochs+1) :
    epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer, device)
    loss_history.append(epoch_loss)
    
    if epoch % 10 == 0 :
        print(f'Epoch {epoch}/{epochs} : loss = {loss_history[-1]}')
        # save model as .pt
        torch.save(model.state_dict(), 'saved_models/scarf_encoder_modified_statedict_1000epochs_SGD_momentum.pt')

end = time.time()
print(end-start)


# save history
df = pd.DataFrame(columns=['epoch', 'loss'])
df['epoch'] = [i for i in range(epochs)]
df['loss'] = loss_history
df.to_csv('training_csv/scarf_modified_1000ep_SGD_momentum.csv')
