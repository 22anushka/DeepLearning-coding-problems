import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
	if X.shape == 2:
		X.reshape(1, X.shape[0], X.shape[1]) 
	return W_q @ X, W_k @ X, W_v @ X

def self_attention(Q, K, V):
	# Need to transpose for correct dimension in dot product from b, n, l, d_n to b, n, d, l
	QK = Q @ K.transpose(0, 1, 3, 2) / (K.shape[-1] ** 0.5) 
	QK_stable = QK - np.max(QK, axis=-1, keepdims=True)  # Stability trick
	attn = np.exp(QK_stable) / np.sum(np.exp(QK_stable), axis=-1, keepdims=True)

	return attn @ V

def multi_head_attention(Q, K, V, n_heads):
	# n_heads -> the dimensions needs to be sliced into n_heads

	# original shape = (b, l, d)
	# new shape = (b, l, n, d_n) where d_n = d // n
	# raise error if d%n != 0

	# this helped split last dimension for the heads
	b, l, d = Q.shape  # Extract batch size, sequence length, and embedding dim
	assert d % n_heads == 0, "Embedding dimension must be divisible by number of heads"
	d_n = d // n_heads  # Dimension per head

	Q = Q.reshape((b, l, n_heads, d_n))
	K = K.reshape((b, l, n_heads, d_n))
	V = V.reshape((b, l, n_heads, d_n))	

	# swap head dimension and sequence length dimension for attention purposes
	q, k, v = Q.transpose(0, 2, 1, 3), K.transpose(0, 2, 1, 3), V.transpose(0, 2, 1, 3)

	# next is to call attention on each of the heads
	# since the dimenisons have been accounted for, no need of splitting, attention should be able to take care of that
	attn = self_attention(q, k, v)

	# have to combine the attention
	attention = attn.transpose(0, 2, 1, 3).reshape(b, l, d)

	return attention

	
