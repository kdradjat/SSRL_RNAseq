import torch
from byol_pytorch.byol import BYOL
from byol_pytorch.NN import Encoder
from torchvision import models

import sys
sys.path.append('../pytorch-scarf')
from cancerclassification.data import *
from cancerclassification.NN import *

#########

# Load Data
print('Loading Data...')
x_unlabel = read_process_data_TCGA('../data/TCGA/pretrain_data.parquet', '../data/TCGA/label.parquet')
x_unlabel, target = x_unlabel[:,1:], x_unlabel[:,0]
nb_classes = len(np.unique(target))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define MLP
encoder = Encoder(input_dim=x_unlabel.shape[1], output_dim=nb_classes, dropout_rate=0.0).to(device)
#encoder = MLP_head(input_dim, emb_dim=nb_classes, l1=1024, l2=512, l3=256,l4=128)

# Define learner
learner = BYOL(
    encoder,
    num_features = x_unlabel.shape[1],
    hidden_layer = -2
)


# set dataloaders
dataset = CancerDatasetTCGA(x_unlabel, target, device=device)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=False)


opt = torch.optim.Adam(learner.parameters(), lr=1e-4)

# csv history
loss_history = []
df = pd.DataFrame(columns=['epoch', 'loss'])

for epoch in range(200):
    running_loss = 0
    print(f'Epoch {epoch}')
    for batch in dataloader :
        x, _ = batch
        loss = learner(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()
        
        running_loss += loss.item() 
    print(f'Initial encoder parameters : {next(encoder.parameters())}')
    
    # save 
    loss_history.append(running_loss / len(dataset))
    df.loc[len(df)] = [epoch, loss_history[-1]]
    df.to_csv('MLP_baseline_improving.csv')
    
# save model
torch.save(encoder.state_dict(), './MLP_baseline_improved.pt')
