import numpy as np

# numpy version
def batch_norm(x, running_mean=None, running_var=None, gamma=None, beta=None, eps=1e-5, momentum=0.1, training=True):
    # Batch Normalization
    # Input: x - input tensor of shape [batch_size, features, ...]
    # Output: normalized tensor of the same shape
    # running_mean, running_var: running statistics for inference
    # momentum: parameter for running statistics update
    # training: whether in training mode or inference mode
    
    # TODO: Implement batch normalization
    # like layer norm but is done over a batch
    
    if running_mean is None:
      running_mean = np.zeros_like(x.shape[0])
    if running_var is None:
      running_var = np.ones_like(x.shape[0])

    if gamma is None:
        gamma = np.ones(x.shape)
    if beta is None:
        beta = np.zeros(x.shape)

    # mean
    mu = np.mean(x, axis=0, keepdims=True)
    # variance
    var = np.sum((x-mu)**2, axis=0, keepdims=True)/(x.shape[0])

    # update_rule
    updated_running_mean = momentum * running_mean + (1-momentum)*mu
    updated_running_var = momentum*running_var + (1-momentum)*var

    normalized_x = ((x-mu)/np.sqrt(var+eps))*gamma + beta
    
    if training:
        # Return normalized tensor and updated running statistics
        return normalized_x, updated_running_mean, updated_running_var
    else:
        # Return only normalized tensor
        return normalized_x
      
