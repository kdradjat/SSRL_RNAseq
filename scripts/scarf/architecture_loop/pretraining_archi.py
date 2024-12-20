import sys

import os

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

import argparse

#################


def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        required=True,
        help="The pretraining data file. Must be in .parquet format."
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="batch size"
    )
    parser.add_argument(
        "--epoch",
        default=100,
        type=int,
        help="number of epoch"
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="dropout rate"
    )
    parser.add_argument(
        "--model_name",
        default="scarf",
        type=str,
        help="name given to the pretrained model"
    )
    parser.add_argument(
        "--history_name",
        default="scarf",
        type=str,
        help="name of the loss history file"
    )
    parser.add_argument(
        "--corruption_rate",
        default=0.3,
        type=float,
        help="corruption rate"
    )
    
    args = parser.parse_args()
    

    # Load data
    x_unlabel = read_process_data_TCGA_unlabel(args.data_file)
    #x_unlabel = read_process_data_TCGA_unlabel('../data/TCGA/100Best_pretrain_data.parquet.gzip')
    #x_unlabel = x_unlabel[:,1:]


    # Training
    corruption_rate = args.corruption_rate
    dropout = args.dropout
    batch_size = args.batch_size # default 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Get validation set
    x_unlabel, x_unlabel_valid = train_test_split(x_unlabel, test_size=0.01, random_state=42)

    train_loader = DataLoader(x_unlabel, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(x_unlabel_valid, batch_size=batch_size, shuffle=True)
    
    # Define embedding size and number of layer lists
    emb_size_list = [256, 1024]
    depth_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for emb_size in emb_size_list :
        for depth in depth_list :
            model = SCARF_Custom(
                input_dim=x_unlabel.shape[1],
                emb_dim=emb_size, 
                encoder_depth=depth,
                features_low=x_unlabel.min(axis=0),
                features_high=x_unlabel.max(axis=0),
                corruption_rate=corruption_rate,
                dropout=dropout
            ).to(device)
            
            import time
            epochs = args.epoch

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
                    torch.save(model.state_dict(), f'saved_models/{args.model_name}_{emb_size}_{depth}.pt')

            end = time.time()
            print(end-start)


            # save history
            df = pd.DataFrame(columns=['epoch', 'loss', 'valid_loss', 'train_loss'])
            df['epoch'] = [i for i in range(epochs)]
            df['loss'] = loss_history
            df['valid_loss'] = valid_loss_history
            df['train_loss'] = train_loss_history
            df.to_csv(f'training_csv/{args.history_name}_{emb_size}_{depth}.csv')
    
    
if __name__ == '__main__' :
    main()
