import torch.nn as nn
import torch.nn.functional as F
import torch
#import wandb
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score




class Encoder(nn.Module):
    """Baseline NN.
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
    
    

    
    
    
    
class MLP_head(nn.Module) :
    def __init__(self, input_dim, output_dim, l1=512, l2=256, l3=128,l4=64, bn=True, dropout_rate=0):
        super().__init__()
        
        self.bn = bn
        
        """self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc1_bn = nn.BatchNorm1d(l1)
        self.dropout2 = nn.Dropout(dropout_rate)"""
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc1_bn = nn.BatchNorm1d(l1)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(l1, l2)
        self.fc2_bn = nn.BatchNorm1d(l2)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(l2, l3)
        self.fc3_bn = nn.BatchNorm1d(l3)
        #self.fc4 = nn.Linear(l3, output_dim)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(l3, l4)
        self.fc4_bn = nn.BatchNorm1d(l4)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(l4, output_dim)
    
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
        if self.bn : x = self.fc3_bn(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc4(x)
        if self.bn : x = self.fc4_bn(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.fc5(x)
        
        return x