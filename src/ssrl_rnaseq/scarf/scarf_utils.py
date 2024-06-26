import random

import numpy as np
import torch
from tqdm.auto import tqdm


def train_epoch(model, criterion, train_loader, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for x in train_loader:
        x = x.to(device)

        # get embeddings
        emb_anchor, emb_positive = model(x)

        # compute loss
        loss = criterion(emb_anchor, emb_positive)
        loss.backward()

        # update model weights
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()

        # log progress
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader.dataset)

def valid_loss_f(model, criterion, train_loader, device) :
    model.eval()
    epoch_loss = 0.0
    
    for x in train_loader : 
        x = x.to(device)
        emb_anchor, emb_positive = model(x)
        loss = criterion(emb_anchor, emb_positive)
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader.dataset)


def dataset_embeddings(model, loader, device):
    embeddings = []

    for x in loader:
        x = x.to(device)
        embeddings.append(model.get_embeddings(x))
    #embeddings = tuple(embeddings)
    embeddings = torch.cat(embeddings, 0).cpu().numpy()

    return embeddings


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

