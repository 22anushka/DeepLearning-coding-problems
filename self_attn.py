import numpy as np

def self_attention(Q, K, V):
    # softmax((Q @ K.T) / sqrt(d_k)) V
    qk = ((Q @ K.T) / K.shape[-1]**0.5)
    # numerical stability
    qk_stable = - np.max(QK, axis=-1, keepdims=True)
    attn = np.exp(qk+qk_stable)/np.sum(exp(qk+qk_stable), axis=-1, keepdims=True)
    attention_output = attn @ V
    return attention_output
