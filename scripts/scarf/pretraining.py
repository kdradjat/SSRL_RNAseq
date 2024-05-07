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

from ssrl_rnaseq.scarf.loss import NTXent, BarlowTwins
from ssrl_rnaseq.scarf.model import SCARF_baseline, MLP, MLP_baseline, SCARF_4layers, SCARF_SubTab
from ssrl_rnaseq.scarf.scarf_utils import *
from ssrl_rnaseq.scarf.dataset import ExampleDataset

from ssrl_rnaseq.data import *
from ssrl_rnaseq.training_utils import *

#################


# Load data
x_unlabel = read_process_data_TCGA_unlabel('../data/TCGA/pretrain_data.parquet')
#x_unlabel = read_process_data_TCGA_unlabel('../data/TCGA/100Best_pretrain_data.parquet.gzip')
#x_unlabel = x_unlabel[:,1:]


# Training
corruption_rate = 0.3
dropout = 0.1
batch_size = 256 # default 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Get validation set
x_unlabel, x_unlabel_valid = train_test_split(x_unlabel, test_size=0.1, random_state=42)

train_loader = DataLoader(x_unlabel, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(x_unlabel_valid, batch_size=batch_size, shuffle=True)

# build model
"""
# 1st architecture
encoder = SCARF_baseline(
    input_dim=pretrain_ds.shape[1],
    emb_dim=100, # 16 by default, augment in the future
    features_low=x_unlabel.min(axis=0),
    features_high=x_unlabel.max(axis=0),
    corruption_rate=corruption_rate,
    dropout=dropout
).to(device)"""

# 2nd architeture
model = SCARF_4layers(
    input_dim=x_unlabel.shape[1],
    emb_dim=256, # 16 by default, augment in the future
    features_low=x_unlabel.min(axis=0),
    features_high=x_unlabel.max(axis=0),
    corruption_rate=corruption_rate,
    dropout=dropout
).to(device)
"""
# 3rd architecture
model = SCARF_SubTab(
    input_dim=x_unlabel.shape[1],
    emb_dim=784, # 16 by default, augment in the future
    features_low=x_unlabel.min(axis=0),
    features_high=x_unlabel.max(axis=0),
    corruption_rate=corruption_rate,
    dropout=dropout
).to(device)
"""

import time
#epochs = 2000
epochs = 1000

optimizer = Adam(model.parameters(), lr = 1e-4)
ntxent_loss = NTXent()


loss_history = []
valid_loss_history = []
train_loss_history = []

start = time.time()
for epoch in range(1, epochs+1) :
    valid_loss = valid_loss_f(model, ntxent_loss, valid_loader, device)
    valid_loss_history.append(valid_loss)
    train_loss = valid_loss_f(model, ntxent_loss, train_loader, device)
    train_loss_history.append(train_loss)
    epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer, device)
    loss_history.append(epoch_loss)
    
    if epoch % 10 == 0 :
        print(f'Epoch {epoch}/{epochs} : loss = {loss_history[-1]}')
        # save model as .pt
        torch.save(model.state_dict(), 'saved_models/scarf_4layers_test.pt')

end = time.time()
print(end-start)


# save history
df = pd.DataFrame(columns=['epoch', 'loss', 'valid_loss', 'train_loss'])
df['epoch'] = [i for i in range(epochs)]
df['loss'] = loss_history
df['valid_loss'] = valid_loss_history
df['train_loss'] = train_loss_history
df.to_csv('training_csv/scarf_4layers_test.csv')
