"""
Computing log probabilities directly is advantageous in numerical computations, especially when dealing with very small probability values, as it enhances numerical stability and prevents underflow issues. ​
The expression logits - np.log(np.sum(np.exp(logits))) computes the log probabilities directly from the logits without first computing the probabilities. This approach is known as the log-softmax function. It is particularly useful because it avoids the potential numerical instability that can arise from computing the softmax probabilities and then taking their logarithm. 
"""

import numpy as np

def log_softmax(logits):
    max_logit = np.max(logits)
    stabilized_logits = logits - max_logit
    log_sum_exp = np.log(np.sum(np.exp(stabilized_logits)))
    return stabilized_logits - log_sum_exp
