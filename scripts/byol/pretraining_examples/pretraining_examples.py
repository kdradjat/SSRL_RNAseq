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
        "--label_file",
        default=None,
        required=True,
        type=str,
        help="The pretraining label file path. Use to stratify pretraining set with incremental. Must be in .parquet format"
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
    x_unlabel_source = read_process_data_ARCHS4_unlabel(args.data_file)
    #nb_classes = len(np.unique(target))
    nb_classes = args.nb_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrain_prop_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    for pretrain_prop in pretrain_prop_list:
        # Get pretraining prop
        #x_unlabel, _ = train_test_split(x_unlabel_source, train_size=pretrain_prop, random_state=42)
        train_idx = generate_indices_pretraining(x_unlabel_source, prop=pretrain_prop)
        x_unlabel = x_unlabel_source[train_idx]
        x_unlabel = x_unlabel[:,1:]

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
            df.to_csv(f'{args.history_name}_{pretrain_prop}.csv')

            # save model
            if (epoch % 5) == 0 :
                torch.save(encoder.state_dict(), f'./{args.model_name}_{pretrain_prop}.pt')
    
if __name__ == '__main__' :
    main()
