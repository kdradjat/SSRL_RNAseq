import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
  

from sklearn.model_selection import train_test_split

from ssrl_rnaseq.data import *
from ssrl_rnaseq.training_utils import *
from ssrl_rnaseq.vime.vime_self import vime_self, vime_self_modified, DAE
from ssrl_rnaseq.vime.vime_utils import perf_metric
from ssrl_rnaseq.vime.supervised_models import *
from ssrl_rnaseq.vime.unfreeze import wrapper, wrapper_rework

from keras.models import load_model

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--data_file",
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
        help="The finetuning label file. Must be in .parquet format."
    )
    parser.add_argument(
        "--batch_size",
        default=8,
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
        "--activation",
        defalut="relu",
        type=str,
        help="activation function type"
    )
    parser.add_argument(
        "--coding_genes",
        default=False,
        type=bool,
        help="take only coding genes"
    )
    parser.add_argument(
        "--preprocess_type",
        default="minmax",
        type=str,
        help="type of data preprocessing: standard or minmax"
    )
    parser.add_argument(
        "--history_name",
        default=None,
        required=True,
        type=str,
        help="name of the accuracy history file"
    )
    
    args = parser.parse_args()

    
    
    # load data
    print('Loading Data...')
    dataset = read_process_data_TCGA(args.data_file, args.label_file, coding_genes=args.coding_genes, preprocess_type=args.preprocess_type)

    emb_size_list = [512, 1024]
    depth_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for emb_size in emb_size_list :
        for depth in depth_list :
            # Finetuning
            prop_list = np.arange(0.02, 1.01, 0.01)
            csv_history = pd.DataFrame(columns=['test_acc', 'prop'])

            # MLP
            mlp_parameters = dict()
            mlp_parameters['epochs'] = args.epoch
            mlp_parameters['activation'] = args.activation
            mlp_parameters['batch_size'] = args.batch_size

            metric = 'acc'

            results_pretraining = []

            for prop in prop_list :
                for i in range(3) :
                    print(prop)
                    # split data
                    idx = generate_indices(dataset, prop=prop, val_prop = 0.01, test_prop = 0.5, rs=42)
                    dataset_train = dataset[idx[0]]
                    dataset_test = dataset[idx[2]]
                    x_train, y_train = dataset_train[:,1:], dataset_train[:,0]
                    x_test, y_test = dataset_test[:,1:], dataset_test[:,0]

                    # load model
                    vime_self_encoder = load_model(f'./saved_models/vime_{emb_size}_{depth}.h5')

                    # finetune model with mlp head
                    y_test_hat = wrapper_rework(vime_self_encoder, x_train, y_train, x_test, mlp_parameters)
                    results_pretraining.append(perf_metric(metric, y_test, y_test_hat))

                    # save 
                    csv_history.loc[len(csv_history)] = [results_pretraining[-1], prop]
                csv_history.to_csv(f'csv_history/finetuning_loop/unfreeze_vime_{emb_size}_{depth}.csv')

if __name__ == "__main__" :
    main()