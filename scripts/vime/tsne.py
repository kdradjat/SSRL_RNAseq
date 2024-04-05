import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
  
from data_loader import load_mnist_data
#from supervised_models import logit, xgb_model, mlp
#from sklearn.model_selection import train_test_split

from ssrl_rnaseq.vime.vime_self import vime_self, vime_self_modified, DAE
from ssrl_rnaseq.vime.vime_semi import vime_semi
from ssrl_rnaseq.vime.vime_utils import perf_metric
from ssrl_rnaseq.vime.utils import *
from keras.models import load_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# load data
x_train = read_process_data('../../data/TCGA/train_data.parquet', '../../data/TCGA/label.parquet')
y_label = x_train[:,0]
x_train = x_train[:,1:]

# load model
vime_self_encoder = load_model('./saved_models/vime_encoder_modified_pm03_20epochs_newoptim.h5')
#vime_self_encoder = load_model('./saved_models/DAE_20epochs.h5')

# get embeddings 
embeddings_train = vime_self_encoder.predict(x_train)

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(embeddings_train)
positive_list = []
for i in range(len(np.unique(y_label))) :
    positive_list.append(y_label == i)

fig, ax = plt.subplots(figsize=(15, 15))

for positive in positive_list :
    ax.scatter(reduced[positive, 0], reduced[positive, 1])
#ax.scatter(reduced[~positive, 0], reduced[~positive, 1], label="negative")
plt.title('tSNE of embeddings produced by VIME encoder (new optimizer)')
plt.legend()

plt.savefig('vime_2000epochs_newoptim.png')
