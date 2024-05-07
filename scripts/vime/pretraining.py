import numpy as np
import pandas as pd
import os
import warnings

from sklearn.model_selection import train_test_split

from ssrl_rnaseq.data import *
from ssrl_rnaseq.training_utils import *
from ssrl_rnaseq.vime.vime_self import vime_self, vime_self_modified, DAE
from ssrl_rnaseq.vime.vime_utils import perf_metric

import matplotlib.pyplot as plt
import keras as keras

# Load Data
print('Loading Data...')
x_unlabel = read_process_data_TCGA('../data/TCGA/pretrain_data.parquet', '../data/TCGA/label.parquet')
#x_unlabel = read_process_data_ARCHS4('../data/ARCHS4/specific/pretrain_specific_data.parquet.gzip', '../data/ARCHS4/specific/pretrain_specific_metadata.parquet.gzip')
#x_unlabel = read_process_data('../data/TCGA/100Best_pretrain_data.parquet.gzip', '../data/TCGA/label.parquet')
y_unlabel = x_unlabel[:,0]
x_unlabel = x_unlabel[:,1:]



# Define Hyper-parameters
p_m = 0.3 # mask proportion : default 0.3
alpha = 2.0

# Metric
metric = 'acc'

# pre-training
vime_self_parameters = {}
vime_self_parameters['batch_size'] = 16
vime_self_parameters['epochs'] = 1000
#vime_self_encoder, history_mask = vime_self_baseline(x_unlabel, p_m, alpha, vime_self_parameters)
vime_self_encoder, history_mask = vime_self_4layers(x_unlabel, p_m, alpha, vime_self_parameters)
#vime_self_encoder, history_mask = vime_self_subtab(x_unlabel, p_m, alpha, vime_self_parameters)
print(history_mask.history.keys())

if not os.path.exists('saved_models') :
    os.makedirs('saved_models')
    
file_name = './saved_models/vime_4layers.h5'

vime_self_encoder.save(file_name)


# save history as csv
df = pd.DataFrame(columns=['epoch', 'loss', 'mask_loss', 'feature_loss', 'val_loss', 'val_mask_loss', 'val_feature_loss'])
df['epoch'] = [i for i in range(vime_self_parameters['epochs'])]
df['loss'] = history_mask.history['loss']
df['mask_loss'] = history_mask.history['mask_loss']
df['feature_loss'] = history_mask.history['feature_loss']
df['val_loss'] = history_mask.history['val_loss']
df['val_mask_loss'] = history_mask.history['val_mask_loss']
df['val_feature_loss'] = history_mask.history['val_feature_loss']
df.to_csv('history_pretraining_vime_4layers.csv')



