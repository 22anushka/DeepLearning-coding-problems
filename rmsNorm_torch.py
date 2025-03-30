import torch
import torch.nn as nn

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        # RMS Normalization
        # Input: x - input tensor of shape [batch_size, ..., features]
        # Output: normalized tensor of the same shape
        # eps: small constant for numerical stability

        # like layer norm but simplified and without shifting, just scaling

        # rms_norm = (x/rms(x)) * gamma; rms(x) = sqrt(eps + mean(x^2))
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape)) # basically shape of features / d_model
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        rms = torch.sqrt(self.eps + (x**2).mean(dim=-1, keepdim=True))
        x = x/rms
        if self.elementwise_affine:
            x = x * self.weight
        return x
