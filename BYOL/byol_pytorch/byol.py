import copy
import random
from functools import wraps
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T



# utils functions

def singleton(cache_key) :
    def inner_fn(fn) :
        @wraps(fn)
        def wrapper(self, *args, **kwargs) :
            instance = getattr(self, cache_key)
            if instance is not None : return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module) :
    return next(module.parameters()).device

def set_requires_grad(model, val) :
    for p in model.parameters() :
        p.requires_grad = val
        
def flatten(t) :
    return t.reshape(t.shape[0], -1)
        

# loss 
def loss_fn(x, y) :
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 2 -2 * (x*y).sum(dim=-1)
    return loss


# EMA 
class EMA() :
    def __init__(self, beta) :
        super().__init__()
        self.beta = beta
    def update_average(self, old, new) :
        if old is None : return new
        return old * self.beta + (1 - self.beta) * new
    
def update_moving_average(ema_updater, ma_model, current_model) :
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()) :
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)
        

        
        
# MLP class for projector and predictor
def MLP(dim, projection_size, hidden_size=4096) :
    model = nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )
    return model
    
    
# Net wrapper
class NetWrapper(nn.Module) :
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2) :
        super().__init__()
        self.net = net
        self.layer = layer
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        
        self.hidden = {}
        self.hook_registered = False
        
    def find_layer(self) :
        if type(self.layer) == str :
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int :
            children = [*self.net.children()]
            return children[self.layer]
        return None
    
    def hook(self, _, input, output) :
        device = input[0].device
        self.hidden[device] = flatten(output)
        
    def register_hook(self) :
        layer = self.find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self.hook)
        self.hook_registered = True
        
    @singleton('projector')
    def get_projector(self, hidden) :
        _, dim = hidden.shape
        create_mlp_fn = MLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)
    
    
    def get_representation(self, x) :
        if self.layer == -1 : return self.net(x)
        if not self.hook_registered :
            self.register_hook()
        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()
        
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden
    
    def forward(self, x, return_projection = True) :
        representation = self.get_representation(x)
        if not return_projection : return representation
        
        projector = self.get_projector(representation)
        projection = projector(representation)
        return projection, representation
    
    
    
# BYOL main class 
class BYOL(nn.Module) :
    def __init__(
        self,
        net, 
        num_features, 
        hidden_layer = -2, 
        projection_size = 256, 
        projection_hidden_size = 4096,
        moving_average_decay = 0.9,
        device='cuda'
    ) :
        
        super().__init__()
        self.net = net
        self.device = device
        
        ### data augmentations
        
        
        ###
        self.online_encoder = NetWrapper(
            net, 
            projection_size, 
            projection_hidden_size, 
            layer = hidden_layer
        )
        
        #self.target_encoder = self.online_encoder
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        
        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)
        
        # send mock sample tensor to instantiate singleton parameters
        self.forward(torch.randn(2, num_features, device=device))
        
    @singleton('target_encoder')
    def get_target_encoder(self) :
        print('call get_target_encoder')
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder
    
    def reset_moving_average(self) :
        del self.target_encoder
        self.target_encoder = None
        
    def update_moving_average(self) :
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        
    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ) :
        
        if return_embedding :
            return self.online_encoder(x, return_projection = return_projection)
        
        
        # define augmentations
        # Generate corrupted samples
        m = mask_generator(0.3, x)
        x_tilde = pretext_generator(m, x)
        #x_tilde.to(self.device)
        sample_one, sample_two = x, x_tilde
        #sample_one, sample_two = x, x
        sample_one = sample_one.to(self.device, dtype=torch.float32)
        sample_two = sample_two.to(self.device, dtype=torch.float32)
        sample_one.requires_grad_()
        sample_two.requires_grad_()
        #print(sample_one)
        #print(sample_two)
        #print(sample_one.get_device())
        #print(sample_two.get_device())
        
        samples = torch.cat((sample_one, sample_two), dim=0)
        print(samples.shape)
        
        online_projections, _ = self.online_encoder(samples)
        online_predictions = self.online_predictor(online_projections)
        
        online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0)
        online_pred_one.requires_grad_()
        online_pred_two.requires_grad_()
        
        with torch.no_grad() :
            #self.target_encoder = self.online_encoder
            #print(self.target_encoder)
            target_encoder = self.get_target_encoder()
            #self.target_encoder = self.online_encoder
            
            #target_projections, _ = self.target_encoder(samples)
            target_projections, _ = target_encoder(samples)
            target_projections = target_projections.detach()
            
            target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)
            target_proj_one.requires_grad_()
            target_proj_two.requires_grad_()
            
            loss_one = loss_fn(online_pred_one, target_proj_two.detach())
            loss_two = loss_fn(online_pred_two, target_proj_one.detach())
            #loss_one.requires_grad_()
            #loss_two.requires_grad_()
            
        loss =  loss_one + loss_two 
            
        return loss.mean()
    
    
    
    
#%%
def mask_generator(p_m, x):
    mask = np.random.binomial(1, p_m, x.shape)
    return mask

#%%
def pretext_generator(m, x):  
      # Parameters
    no, dim = x.shape
    x_copy = x.cpu()
      # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x_copy[idx, i]

      # gaussian noise vector
    x_noise = np.zeros([no, dim])
    x_noise = np.random.normal(0, 1, size=(no, dim))

      # Corrupt samples
    #x_tilde = x * (1-m) + (x + x_noise) * m  
    x_tilde = x_copy * (1-m) + x_bar * m
    

    return x_tilde
        
    
    
    
    
    