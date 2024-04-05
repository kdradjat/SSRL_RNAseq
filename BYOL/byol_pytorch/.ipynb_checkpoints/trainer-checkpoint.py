from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn import SyncBatchNorm

from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader

from byol_pytorch.byol_pytorch import BYOL

from beartype import beartype
from beartype.typing import Optional


############

def exists(a) :
    return a is not None

def cycle(dl) :
    while True :
        for batch in dl :
            yield batch
            

# class

class BYOLTrainer(Module) :
    @beartype
    def __init__(
        self, 
        net : Module,
        *,
        image_size : int, 
        hidden_layer : str,
        learning_rate : float,
        dataset : Dataset,
        num_train_steps : int,
        batch_size : int=128,
        optimizer_klass = Adam,
        checkpoint_every : int=1000,
        checkpoint_folder : str='./checkpoints',
        byol_kwargs : dict = dict(),
        optimizer_kwargs : dict = dict()
    ):
        
        
        super().__init__()
        
        self.net = net
        self.byol = BYOL(net, image_size=image_size, hidden_layer=hidden_layer, **byol_kwargs)
        self.optimizer = optimizer_klass(self.byol.parameters(), lr = learning_rate, **optimizer_kwargs)
        self