import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
  
#from data_loader import load_mnist_data
from supervised_models import logit, mlp
from sklearn.model_selection import train_test_split

from ssrl_rnaseq.vime.vime_self import vime_self, vime_self_modified, DAE
from ssrl_rnaseq.vime.vime_semi import vime_semi
from ssrl_rnaseq.vime.vime_utils import perf_metric
from ssrl_rnaseq.vime.utils import *
from keras.models import load_model


# load data
print('Loading Data...')
dataset = read_process_data('../../data/TCGA/nopretrain_data.parquet', '../../data/TCGA/label.parquet')


# Finetuning
from scipy.special import softmax

prop_list = np.arange(0.02, 1.0, 0.01)
csv_history = pd.DataFrame(columns=['test_acc', 'prop'])

# MLP
mlp_parameters = dict()
mlp_parameters['hidden_dim'] = 100
mlp_parameters['epochs'] = 100
mlp_parameters['activation'] = 'relu'
mlp_parameters['batch_size'] = 32

metric = 'acc'

results_pretraining = []

for prop in prop_list :
    for i in range(5) :
        print(prop)
        # split data
        idx = generate_indices(dataset, prop=prop, val_prop = 0.1, test_prop = 0.5, rs=42)
        dataset_train = dataset[idx[0]]
        dataset_test = dataset[idx[2]]
        x_train, y_train = dataset_train[:,1:], dataset_train[:,0]
        x_test, y_test = dataset_test[:,1:], dataset_test[:,0]

        # load model
        vime_self_encoder = load_model('./saved_models/vime_encoder_modified_pm03_20epochs.h5')
        #vime_self_encoder = load_model('./saved_models/DAE_20epochs.h5')
        # get embeddings
        x_train_hat = vime_self_encoder.predict(x_train)
        x_test_hat = vime_self_encoder.predict(x_test)

        # finetune mlp
        #y_test_hat = mlp(x_train_hat, y_train, x_test_hat, mlp_parameters)
        y_test_hat = logit(x_train_hat, y_train, x_test_hat)
        print(len(np.argmax(y_test_hat, axis=1)))
        results_pretraining.append(perf_metric(metric, y_test, y_test_hat))
        print(results_pretraining[-1])

        # save 
        csv_history.loc[len(csv_history)] = [results_pretraining[-1], prop]
    csv_history.to_csv('vime_20epochs_logreg.csv')





