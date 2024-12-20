import torch
from ssrl_rnaseq.byol.byol import BYOL
from ssrl_rnaseq.byol.NN import Encoder, MLP_head, _4layers, baseline, _Nlayers
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
        help="The pretraining data file path. Must be in .parquet format"
    )
    parser.add_argument(
        "--nb_classes",
        default=19,
        type=int,
        help="number of classes wanted"
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="batch size"
    )
    parser.add_argument(
        "--epoch",
        default=200,
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
        default="byol_history",
        help="name of the loss history file"
    )
    args = parser.parse_args()

    
    # Load Data
    print('Loading Data...')
    #x_unlabel = read_process_data_TCGA('../data/TCGA/pretrain_data.parquet', '../data/TCGA/label.parquet')
    #x_unlabel = read_process_data_TCGA('../data/TCGA/100Best_pretrain_data.parquet.gzip', '../data/TCGA/label.parquet')
    #x_unlabel, target = x_unlabel[:,1:], x_unlabel[:,0]
    x_unlabel = read_process_data_TCGA_unlabel(args.data_file)
    #nb_classes = len(np.unique(target))
    nb_classes = args.nb_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # Define MLP
    #encoder = Encoder(input_dim=x_unlabel.shape[1], output_dim=nb_classes, dropout_rate=0.0).to(device)
    #encoder = MLP_head(input_dim=x_unlabel.shape[1], output_dim=nb_classes, l1=1024, l2=512, l3=256,l4=128).to(device)
    encoder = _4layers(input_dim=x_unlabel.shape[1], output_dim=nb_classes).to(device)

    # Define learner
    learner = BYOL(
        encoder,
        num_features = x_unlabel.shape[1],
        hidden_layer = -2
    )


    # set dataloaders
    dataset = CancerDatasetTCGA(x_unlabel, target, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


    opt = torch.optim.Adam(learner.parameters(), lr=1e-4)

    # csv history
    loss_history = []
    df = pd.DataFrame(columns=['epoch', 'loss'])

    for epoch in range(args.epoch):
        running_loss = 0
        print(f'Epoch {epoch}')
        for batch in dataloader :
            x = batch
            loss = learner(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
        
            running_loss += loss.item() 
        #print(f'Initial encoder parameters : {next(encoder.parameters())}')
    
        # save 
        loss_history.append(running_loss / len(dataset))
        df.loc[len(df)] = [epoch, loss_history[-1]]
        df.to_csv(f'{args.model_name}.csv')
    
    # save model
    torch.save(encoder.state_dict(), f'./{history_name}.pt')
    
if __name__ == '__main__' :
    main()
