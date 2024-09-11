import numpy as np
import pandas as pd
import os
import warnings

from sklearn.model_selection import train_test_split

from ssrl_rnaseq.data import *
from ssrl_rnaseq.training_utils import *
from ssrl_rnaseq.vime.vime_self import vime_self, vime_self_modified, DAE, vime_self_4layers, vime_self_subtab, vime_self_custom, dae
from ssrl_rnaseq.vime.vime_utils import perf_metric

import keras as keras

import argparse



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
        default=16,
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
        "--history_name",
        default="history_vime.csv",
        type=str,
        help="name of the loss history file"
    )
    parser.add_argument(
        "--mask_prop",
        type=float,
        default=0.3,
        help="mask proportion"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="alpha parameters"
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
    
    args = parser.parse_args()


    
    # Load Data
    print('Loading Data...')
    x_unlabel_source = read_process_data_TCGA_unlabel(args.data_file, coding_genes=args.coding_genes, preprocess_type=args.preprocess_type)
    
    pretrain_prop_list = np.arange(0.1, 1.0, 0.05)
    
    for pretrain_prop in pretrain_prop_list:
        # Get pretraining prop
        x_unlabel, _ = train_test_split(x_unlabel_source, train_size=pretrain_prop, random_state=0)    

        # Define Hyper-parameters
        p_m = args.mask_prop # mask proportion : default 0.3
        alpha = args.alpha

        # Metric
        metric = 'acc'

        # pre-training
        vime_self_parameters = {}
        vime_self_parameters['batch_size'] = args.batch_size
        vime_self_parameters['epochs'] = args.epoch
        #vime_self_encoder, history_mask = vime_self_baseline(x_unlabel, p_m, alpha, vime_self_parameters)
        vime_self_encoder, history_mask = vime_self_4layers(x_unlabel, p_m, alpha, vime_self_parameters)
        #vime_self_encoder, history_mask = vime_self_subtab(x_unlabel, p_m, alpha, vime_self_parameters)
        #vime_self_encoder, history_mask = dae(x_unlabel, p_m, alpha, vime_self_parameters)
        print(history_mask.history.keys())

        if not os.path.exists('saved_models') :
            os.makedirs('saved_models')
    
        file_name = f'./saved_models/vime_pretrainprop_{pretrain_prop}.h5'

        vime_self_encoder.save(file_name)
    
        df = pd.DataFrame(columns=['epoch', 'loss', 'mask_loss', 'feature_loss', 'val_loss', 'val_mask_loss', 'val_feature_loss'])
        df['epoch'] = [i for i in range(vime_self_parameters['epochs'])]
        df['loss'] = history_mask.history['loss']
        df['mask_loss'] = history_mask.history['mask_loss']
        df['feature_loss'] = history_mask.history['feature_loss']
        df['val_loss'] = history_mask.history['val_loss']
        df['val_mask_loss'] = history_mask.history['val_mask_loss']
        df['val_feature_loss'] = history_mask.history['val_feature_loss']
        df.to_csv(args.history_name)
        
if __name__ == '__main__' :
    main()
