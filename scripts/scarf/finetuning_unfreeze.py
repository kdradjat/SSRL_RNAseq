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

from ssrl_rnaseq.scarf.loss import NTXent
from ssrl_rnaseq.scarf.model import SCARF, SCARF_baseline, MLP, MLP_head_simple, LastLayer, SCARF_SubTab, SCARF_4layers
from ssrl_rnaseq.scarf.dataset import ExampleDataset
from ssrl_rnaseq.scarf.scarf_utils import *

from ssrl_rnaseq.data import *
from ssrl_rnaseq.training_utils import *

import argparse

##############

def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--pretraining_file",
        default=None,
        type=str,
        required=True,
        help="The pretraining data file. Must be in .parquet format."
    )
    parser.add_argument(
        "--finetuning_file",
        default=None,
        type=str,
        required=True,
        help="The finetuning data file. Must be in .parquet format."
    )
    parser.add_argument(
        "--label_file",
        default=None,
        type=str,
        required=True,
        help="The label data file. Must be in .parquet format."
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="batch size"
    )
    parser.add_argument(
        "--corruption_rate",
        default=0.3,
        type=float,
        help="corruption rate"
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="dropout rate"
    )
    parser.add_argument(
        "--model_name",
        default=None,
        required=True,
        type=str,
        help="name of the pretrained model"
    )
    parser.add_argument(
        "--history_name",
        default=None,
        required=True,
        type=str,
        help="name of the accuracy history file"
    )
    parser.add_argument(
        "--epoch",
        defalut=100,
        type=int,
        help="number of epoch"
    )
    args = parser.parse_args()

    # Load Data
    print('Loading Data...')
    dataset_pretrain = read_process_data_TCGA(args.pretraining_file, args.label_file)
    dataset = read_process_data_TCGA(args.finetuning_file, args.label_file)
    #dataset_pretrain = read_process_data_ARCHS4('../data/ARCHS4/specific/pretrain_specific_data.parquet.gzip', '../data/ARCHS4/specific/pretrain_specific_metadata.parquet.gzip')
    #dataset = read_process_data_ARCHS4('../data/ARCHS4/specific/nopretrain_specific_data.parquet.gzip', '../data/ARCHS4/specific/nopretrain_specific_metadata.parquet.gzip')
    nb_classes = len(np.unique(dataset[:,0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrain_ds = ExampleDataset(
        dataset_pretrain[:,1:],
        dataset_pretrain[:,0]
    )

    # Training
    name = f"{args.history_name}"
    logger = LogResults(name, ["prop"])
    bs = args.batch_size
    prop_list = np.arange(0.02, 1.0, 0.01)
    corruption_rate = args.corruption_rate
    dropout = args.dropout


    for i in range(3) :
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

            pretrained_model.load_state_dict(torch.load(f"{args.model_name}"), strict=False)
            # isolate encoder part
            encoder = pretrained_model.encoder

            # split training set depending on prop value
            idx = generate_indices(dataset, prop=prop, val_prop=0.01, test_prop=0.5, rs=42)

            # split 
            embeddings_dataset = CancerDatasetTCGA(dataset[:,1:], dataset[:,0], device)
            dataloaders = get_dataloaders(embeddings_dataset, idx, [bs, bs, bs])

            # build MLP
            config_nn = {
                "epochs":args.epoch,
                "lr_init":1e-4,
                "early_stop":30,
                "optim":optim.Adam,
                "bn":True,
                "dropout_rate":0.0
            }
        
            head = LastLayer(input_dim=256, output_dim=nb_classes).to(device)
        
            # Merge encoder and MLP
            model = nn.Sequential(encoder, head)
        
            train_nn(config_nn, dataloaders, model, weights=None, early_stop=config_nn["early_stop"], log=False, logger=logger, test=True, mlp_only=True)
            logger.next_run()
            logger.show_progression()

        logger.save_csv()
        
if __name__ == "__main__" :
    main()

