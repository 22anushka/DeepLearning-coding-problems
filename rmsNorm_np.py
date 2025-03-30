import numpy as np

def rms_norm(x, gamma=None, eps=1e-8):
    # RMS Normalization
    # Input: x - input tensor of shape [batch_size, ..., features]
    # Output: normalized tensor of the same shape
    # eps: small constant for numerical stability
    
    # TODO: Implement RMS normalization
    # like layer norm but simplified and without shifting, just scaling

    # rms_norm = (x/rms(x)) * gamma; rms(x) = sqrt(eps + mean(x^2))
    
    if gamma is None:
      gamma = np.ones(x.shape)

    rms_x = np.sqrt(eps + np.mean(x**2, axis=-1, keepdims=True))

    normalized_x = (x/rms_x)*gamma
    
    return normalized_x
