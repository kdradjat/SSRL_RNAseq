import torch.nn as nn
import torch.nn.functional as F
import torch
#import wandb
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score




class Encoder(nn.Module):
    """Basic neural network for cancer classification task.

    Args:
        input_dim (int): Inputs dimension.
        output_dim (int): Outputs dimension.
        l1 (int, optional): Dimension of the first hidden layer. Defaults to 512.
        l2 (int, optional): Dimension of the second hidden layer. Defaults to 256.
        l3 (int, optional): Dimension of the third hidden layer. Defaults to 128.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=100, bn=True, dropout_rate=0):
        super().__init__()
        
        self.bn = bn
        
        """self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc1_bn = nn.BatchNorm1d(l1)
        self.dropout2 = nn.Dropout(dropout_rate)"""
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        if self.bn : x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.bn : x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = F.softmax(x)
        
        return x