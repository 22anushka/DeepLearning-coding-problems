import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        # Batch Normalization
        # Input: x - input tensor of shape [batch_size, features, ...]
        # Output: normalized tensor of the same shape
        # normalized_shape (int): Number of channels/features in the input
        # eps (float): Small constant for numerical stability
        # momentum (float): Momentum factor for updating running stats
        # affine (bool): If True, includes learnable scale and shift (gamma & beta)
        # track_running_stats (bool): If True, tracks running mean/var

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = affine
        self.momentum = momentum
        self.running_mean = torch.nn.zeros(self.normalized_shape)
        self.running_var = torch.nn.ones(self.normalized_shape)
        self.track_running_stats = track_running_stats
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape)) # basically shape of features / d_model
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mu = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True)
        x = ((x-mu)/torch.sqrt(var+self.eps))

        if self.elementwise_affine:
          x = x*self.weight + self.bias

        # idk if squeeze is to be there or not
        if self.track_running_stats:
          self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*mu.squeeze()
          self.running_var = self.momentum*self.running_var + (1-self.momentum)*var.squeeze()
          return x, self.running_mean, self.running_var
        
        return x

