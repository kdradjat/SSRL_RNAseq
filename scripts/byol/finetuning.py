import torch
from ssrl_rnaseq.byol.byol import BYOL
from ssrl_rnaseq.byol.NN import Encoder, MLP_head, _4layers, baseline
from torchvision import models

from ssrl_rnaseq.data import *
from ssrl_rnaseq.NN import *

import argparse

#########

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_file",
        default=None,
        required=True,
        type=str,
        help="The finetuning data file path. Must be in .parquet format"
    )
    parser.add_argument(
        "--label_file",
        default=None,
        required=True,
        type=str,
        help="The finetuning label file path. Must be in .parquet format"
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
        "--model_name",
        required=True,
        type=str,
        help="name of the saved model"
    )
    parser.add_argument(
        "--history_name",
        type=str,
        default="byol_history_finetuning",
        help="name of the accuracy history file"
    )
    args = parser.parse_args()


    # Load Data
    print('Loading Data...')
    dataset = read_process_data_TCGA(args.data_file, args.label_file)
    nb_classes = len(np.unique(dataset[:,0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




    # Training
    name = args.history_name
    logger = LogResults(name, ["prop"])
    bs = args.batch_size
    prop_list = np.arange(0.02, 1.01, 0.01)

    for i in range(5) :
        for prop in prop_list :
            # logger
            logger.update_hyps([prop])
        
            # Build and Load model
            #encoder = MLP_head(input_dim=dataset[:,1:].shape[1], output_dim=nb_classes, l1=1024, l2=512, l3=256,l4=128, dropout_rate=0.0).to(device)
            encoder = Encoder(input_dim=dataset[:,1:].shape[1], output_dim=nb_classes, dropout_rate=0.0).to(device)
            #encoder.load_state_dict(torch.load("MLP_baseline_improved.pt"), strict=False)
            encoder.load_state_dict(torch.load(f"{model_name}"), strict=False)

            # split training set depending on prop value
            idx = generate_indices(dataset, prop=prop, val_prop=0.1, test_prop=0.5, rs=42)

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
        
            # freeze param expect last layer
            # Encoder (simple)
            #for i, param in zip(range(7), encoder.parameters()):
                #param.requires_grad = False
            # MLP head (big)
            #for i, param in zip(range(15), encoder.parameters()):
                #param.requires_grad = False
        
        
            train_nn(config_nn, dataloaders, encoder, weights=None, early_stop=config_nn["early_stop"], log=False, logger=logger, test=True, mlp_only=True)
            logger.next_run()
            logger.show_progression()

        logger.save_csv()
        
        
if __name__ == '__main__' :
    main()
