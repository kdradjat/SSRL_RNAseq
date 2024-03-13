import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
  
#from data_loader import load_mnist_data
#from supervised_models import logit, xgb_model, mlp
from sklearn.model_selection import train_test_split

from vime_self import vime_self, vime_self_modified, DAE
from vime_semi import vime_semi
from vime_utils import perf_metric
from utils import *

import matplotlib.pyplot as plt
import keras as keras

# Load Data
print('Loading Data...')
x_unlabel = read_process_data('../data/TCGA/pretrain_data.parquet', '../data/TCGA/label.parquet')
y_unlabel = x_unlabel[:,0]
x_unlabel = x_unlabel[:,1:]



# Define Hyper-parameters
p_m = 0.3 # mask proportion : default 0.3
alpha = 2.0

# Metric
metric = 'acc'

# pre-training
vime_self_parameters = {}
vime_self_parameters['batch_size'] = 32
vime_self_parameters['epochs'] = 20
vime_self_encoder, history_mask = vime_self_modified(x_unlabel, p_m, alpha, vime_self_parameters)
#vime_self_encoder, history_mask = DAE(x_unlabel, p_m, alpha, vime_self_parameters)

if not os.path.exists('saved_models') :
    os.makedirs('saved_models')
    
file_name = './saved_models/vime_encoder_modified_pm03_20epochs_lr1e-4_MA.h5'
#file_name = './saved_models/DAE_20epochs.h5'

vime_self_encoder.save(file_name)


# save history as csv
df = pd.DataFrame(columns=['epoch', 'loss', 'mask_loss', 'feature_loss'])
df['epoch'] = [i for i in range(vime_self_parameters['epochs'])]
df['loss'] = history_mask.history['loss']
df['mask_loss'] = history_mask.history['mask_loss']
df['feature_loss'] = history_mask.history['feature_loss']
df.to_csv('history_pretraining_vimepm03_2000epochs_newoptim_schedule.csv')


# Plot Loss
plt.figure()
plt.plot(history_mask.history['loss'], label='total_loss')
#plt.plot(history_mask.history['mask_loss'], label='mask_loss')
#plt.plot(history_mask.history['feature_loss'], label='feature_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('losses over epochs')

if not os.path.exists('figures') :
    os.makedirs('figures')

plt.savefig('figures/loss_vime_newarchi_pm03_200epochs.png')
