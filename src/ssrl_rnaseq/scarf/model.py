import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torch.nn.functional as F


class MLP(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dim, num_hidden, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.extend([nn.Linear(in_dim, hidden_dim), nn.Dropout(dropout)])

        super().__init__(*layers)

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
    
class MLP_head_simple(nn.Module) :
    def __init__(self, input_dim, output_dim, bn=True, dropout_rate=0):
        super().__init__()
        
        self.bn = bn
        
        """self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc1_bn = nn.BatchNorm1d(l1)
        self.dropout2 = nn.Dropout(dropout_rate)"""
        
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        if self.bn : x = self.fc1_bn(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        if self.bn : x = self.fc2_bn(x)
        x = self.fc3(x)
        
        return x
"""    
class Encoder(torch.nn.Sequential) :
    def __init__(self, input_dim, output_dim, emb_dim, num_hidden= bn=True, dropout_rate=0):
        self.bn = bn
        layers = []
        for 
"""

class SCARF(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        features_low,
        features_high,
        nb_classes,
        num_hidden=4,
        head_depth=2,
        corruption_rate=0.6,
        dropout=0.0
    ):
        super().__init__()

        self.encoder = MLP(input_dim, emb_dim, num_hidden, dropout)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)
        #self.head = MLP(emb_dim, nb_classes, num_hidden, dropout)
        self.head = MLP_head(emb_dim, nb_classes)
        #self.head = MLP_head_simple(emb_dim, nb_classes)

        # uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_len = int(corruption_rate * input_dim)

    def forward(self, x):
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        embeddings = self.encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        return embeddings, embeddings_corrupted

    def forward_encoder(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
    
    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)
    
    

class SCARF_modified(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        features_low,
        features_high,
        nb_classes,
        num_hidden=4,
        head_depth=2,
        corruption_rate=0.6,
        dropout=0.0
    ):
        super().__init__()

        #self.encoder = MLP(input_dim, emb_dim, num_hidden, dropout)
        self.encoder = MLP_head(input_dim, emb_dim, l1=1024, l2=512, l3=256,l4=128)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)
        #self.head = MLP(emb_dim, nb_classes, num_hidden, dropout)
        self.head = MLP_head(emb_dim, nb_classes)
        #self.head = MLP_head_simple(emb_dim, nb_classes)

        # uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_len = int(corruption_rate * input_dim)

    def forward(self, x):
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        embeddings = self.encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        return embeddings, embeddings_corrupted

    def forward_encoder(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
    
    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)
    
    
    
##################### REWORK #############################
class MLP_builder(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dim, num_hidden, dropout=0.2):
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        super().__init__(*layers)
        
        
class SCARF_Custom(nn.Module) :
    def __init__(
        self,
        input_dim,
        emb_dim,
        encoder_depth,
        features_low,
        features_high,
        head_depth=2,
        corruption_rate=0.6,
        dropout=0.2
    ):
        super().__init__()

        self.encoder = MLP_builder(input_dim, emb_dim, encoder_depth, dropout)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth, dropout)

        # uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_len = int(corruption_rate * input_dim)

    def forward(self, x):
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        embeddings = self.encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        return embeddings, embeddings_corrupted
    
    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)


class MLP_baseline(nn.Module) :
    def __init__(self, input_dim, bn=True):
        super().__init__()
        
        self.bn = bn
        
        """self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc1_bn = nn.BatchNorm1d(l1)
        self.dropout2 = nn.Dropout(dropout_rate)"""
        
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        if self.bn : x = self.fc1_bn(x)
        x = self.fc2(x)
        x = F.relu(x)
        if self.bn : x = self.fc2_bn(x)
        
        return x

class LastLayer(nn.Module) :
    def __init__(self, input_dim, output_dim) :
        super().__init__()
        
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x)
        
        return x
    
class SCARF_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2) :
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        return x
    
class SubTab(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=1024, hidden_dim_2=784):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim_2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_bn(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc2_bn(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc3_bn(x)
        
        return x

class _4layers(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout_rate=0.2) :
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        x = F.softmax(x)
        
        return x
        
    
    
class SCARF_baseline(nn.Module) :
    def __init__(
        self,
        input_dim,
        emb_dim,
        features_low,
        features_high,
        head_depth=2,
        corruption_rate=0.6,
        dropout=0.0
    ):
        super().__init__()

        self.encoder = MLP_baseline(input_dim)
        #self.encoder = SCARF_encoder(input_dim)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        # uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_len = int(corruption_rate * input_dim)

    def forward(self, x):
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        embeddings = self.encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        return embeddings, embeddings_corrupted
    
    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)
    
    
class SCARF_4layers(nn.Module) :
    def __init__(
        self,
        input_dim,
        emb_dim,
        features_low,
        features_high,
        head_depth=2,
        corruption_rate=0.6,
        dropout=0.2
    ):
        super().__init__()

        #self.encoder = MLP_baseline(input_dim)
        self.encoder = SCARF_encoder(input_dim, dropout=dropout)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        # uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_len = int(corruption_rate * input_dim)

    def forward(self, x):
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        embeddings = self.encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        return embeddings, embeddings_corrupted
    
    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)
    

class SCARF_SubTab(nn.Module) :
    def __init__(
        self,
        input_dim,
        emb_dim,
        features_low,
        features_high,
        head_depth=2,
        corruption_rate=0.6,
        dropout=0.0
    ):
        super().__init__()

        self.encoder = SubTab(input_dim)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        # uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_len = int(corruption_rate * input_dim)

    def forward(self, x):
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        embeddings = self.encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        return embeddings, embeddings_corrupted
    
    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)
