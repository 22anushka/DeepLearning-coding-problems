import numpy as np

# numpy version
def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    # Layer Normalization
    # Input: x - input tensor of shape [batch_size, ..., features]
    # Output: normalized tensor of the same shape
    # eps: small constant for numerical stability
    
    # TODO: Implement layer normalization

    # mean of the input_tensor, for each sample in the batch
    # variance of the tensor for each sample in the batch

    # LayerNorm formula -> [(x-mean)/sqrt(variance + eps)]*scale + shift where scale and shift are learnable parameters

    if gamma is None:
        gamma = np.ones(x.shape[-1])
    if beta is None:
        beta = np.zeros(x.shape[-1])

    # mean
    mu = np.mean(x, axis=-1, keepdims=True)
    # variance
    var = np.sum((x-mu)**2, axis=-1, keepdims=True)/(x.shape[-1])
    # can also do np.var(x)
    # var = np.var(x, axis=-1, keepdims=True)

    normalized_x = ((x-mu)/np.sqrt(var+eps))*gamma + beta
    
    return normalized_x
