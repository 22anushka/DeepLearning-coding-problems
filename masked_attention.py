import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d_k = K.shape[-1]
    attn = Q @ K.T / (d_k ** 0.5)
    # assuming we have to create our own mask (even though mask has been provided)
    mask = np.ones_like(attn, dtype=bool)
    mask = np.tril(mask)
    if mask is not None:
        attn = np.where(mask, attn, -np.inf)
    attn = np.exp(attn) / attn.sum(axis=-1, keepdims=True)
    attn = attn @ V
    return attn
