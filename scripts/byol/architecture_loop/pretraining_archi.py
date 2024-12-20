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
        default=20,
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
    x_unlabel = read_process_data_TCGA_unlabel(args.data_file)
    nb_classes = args.nb_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emb_size_list = [256, 1024]
    depth_list = [2,3,4,5,6,7,8,9,10]
    
    for emb_size in emb_size_list :
        for depth in depth_list :
            # Define MLP
            encoder = _Nlayers(input_dim=x_unlabel.shape[1], output_dim=nb_classes).to(device)

            # Define learner
            learner = BYOL(
                encoder,
                num_features = x_unlabel.shape[1],
                hidden_layer = -2
            )


            # set dataloaders
            dataloader = torch.utils.data.DataLoader(x_unlabel, batch_size=args.batch_size, shuffle=False)


            opt = torch.optim.SGD(learner.parameters(), lr=1e-4, momentum=0.9)

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
                df.to_csv(f'{args.history_name}_{depth}_{emb_size}.csv')

            # save model
            torch.save(encoder.state_dict(), f'./{args.model_name}_{depth}_{emb_size}.pt')
    
if __name__ == '__main__' :
    main()
