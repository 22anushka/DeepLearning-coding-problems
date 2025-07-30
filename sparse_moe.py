import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    # use the gating weights to find the top k experts
    logits = x @ Wg 
    stable_logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(stable_logits) / np.sum(np.exp(stable_logits), axis=-1, keepdims=True)

    # selecting only the last k and then reversing (descending order)
    # shape must be b,s,top_k since we need it for every token
    topk_idx = np.argsort(probs, axis=-1)[..., -top_k:][..., ::-1]  # (b,s,top_k)

    # same as a sliced approach
    topk_probs = np.take_along_axis(probs, topk_idx, axis=-1)
    # normalizing the gating probabilities
    topk_probs /= topk_probs.sum(axis=-1, keepdims=True)
    
    
    logits =  np.einsum('b s d, e d h -> b s e h', x, We)
    
    # gathering experts seems to be more efficient than masking experts out

    # gather only the experts
    # this is required because top_k_idx is of shape b,s,k
    # so if we dont use this, the sliced shape would increase
    b_idx = np.arange(x.shape[0])[:, None, None]   # shape (B,1,1)
    s_idx = np.arange(x.shape[1])[None, :, None]
    selected = logits[b_idx,  # picks along axis 0 (batch)
                       s_idx,  # picks along axis 1 (seq)
                       topk_idx,  # picks along axis 2 (experts)
                       :]  # “:” picks the entire last dimension (model dim D)
    out = (selected * topk_probs[...,None]).sum(axis=2)
    return out
