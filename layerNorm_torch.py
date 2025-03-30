import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        # Layer Normalization
        # Input: x - input tensor of shape [batch_size, ..., features]
        # Output: normalized tensor of the same shape
        # eps: small constant for numerical stability
        
        # TODO: Implement layer normalization using torch

        # mean of the input_tensor, for each sample in the batch
        # variance of the tensor for each sample in the batch

        # LayerNorm formula -> [(x-mean)/sqrt(variance + eps)]*scale + shift where scale and shift are learnable parameters
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape)) # basically shape of features / d_model
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x
