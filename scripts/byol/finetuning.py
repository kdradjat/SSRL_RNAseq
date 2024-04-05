import torch
from ssrl_rnaseq.byol.byol import BYOL
from ssrl_rnaseq.byol.NN import Encoder, MLP_head
from torchvision import models

from ssrl_rnaseq.data import *
from ssrl_rnaseq.NN import *

#########

# Load Data
print('Loading Data...')
dataset = read_process_data_TCGA('../data/TCGA/nopretrain_data.parquet', '../data/TCGA/label.parquet')
nb_classes = len(np.unique(dataset[:,0]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
# Define and load MLP
encoder = Encoder(input_dim=x_unlabel.shape[1], output_dim=nb_classes, dropout_rate=0.0).to(device)
#encoder = MLP_head(input_dim, emb_dim=nb_classes, l1=1024, l2=512, l3=256,l4=128)

encoder.load_state_dict(torch.load("MLP_baseline_improved.pt"), strict=False)
"""

# Training
name = "big_freeze"
logger = LogResults(name, ["prop"])
bs = 32
prop_list = np.arange(0.02, 1.01, 0.01)
#prop_list = [1]



for i in range(5) :
    for prop in prop_list :
        # logger
        logger.update_hyps([prop])
        
        # Build and Load model
        encoder = MLP_head(input_dim=dataset[:,1:].shape[1], output_dim=nb_classes, l1=1024, l2=512, l3=256,l4=128, dropout_rate=0.0).to(device)
        #encoder = Encoder(input_dim=dataset[:,1:].shape[1], output_dim=nb_classes, dropout_rate=0.0).to(device)
        #encoder.load_state_dict(torch.load("MLP_baseline_improved.pt"), strict=False)
        encoder.load_state_dict(torch.load("MLP_modified_improved.pt"), strict=False)

        # split training set depending on prop value
        idx = generate_indices(dataset, prop=prop, val_prop=0.1, test_prop=0.5, rs=42)

        # get embeddings
        #loader = DataLoader(dataset[:,1:], batch_size=bs, shuffle=False)
        #embeddings = dataset_embeddings(encoder, loader, device)

        # split 
        embeddings_dataset = CancerDatasetTCGA(dataset[:,1:], dataset[:,0], device)
        dataloaders = get_dataloaders(embeddings_dataset, idx, [bs, bs, bs])

        # build MLP
        config_nn = {
            "epochs":100,
            "lr_init":1e-4,
            "early_stop":30,
            "optim":optim.Adam,
            "bn":True,
            "dropout_rate":0.0
        }
        
        # freeze param expect last layer
        # Encoder (simple)
        #for i, param in zip(range(7), encoder.parameters()):
            #param.requires_grad = False
        # MLP head (big)
        for i, param in zip(range(15), encoder.parameters()):
            param.requires_grad = False
        
        
        train_nn(config_nn, dataloaders, encoder, weights=None, early_stop=config_nn["early_stop"], log=False, logger=logger, test=True, mlp_only=True)
        logger.next_run()
        logger.show_progression()

    logger.save_csv()
