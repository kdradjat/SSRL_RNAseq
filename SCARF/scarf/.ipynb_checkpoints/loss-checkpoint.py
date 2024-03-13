import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXent(nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss
    
    
class BarlowTwins(nn.Module) :
    def __init__(self, lb=5e-3) :
        self.lb = lb
    
    def forward(self, z_a, z_b) :
        batch_size = z_a.size(0) # N
        
        # normalize along the batch dimension
        #z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_a - torch.mean(z_a, 0, True)) / torch.std(z_a, 0, True)
        z_b_norm = (z_b - torch.mean(z_b, 0, True)) / torch.std(z_b, 0, True) # NxD
        
        # cross-correlation matrix
        c = torch.matmul(z_a_norm.T, z_b_norm) / batch_size # DxD
        
        # loss
        eye = torch.eye(z_a.size(1))
        c_diff = torch.linalg.matrix_power(c - eye, 2) # DxD
        c_diff_copy = c_diff.clone()
        c_diff_copy.fill_diagonal_(0)
        c_diff_copy = torch.mul(c_diff_copy, self.lb)
        c_final = torch.diag(c_diff) + c_diff_copy
        loss = torch.sum(c_final)
        
        return loss
        
        
