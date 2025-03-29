"""
Multi-query attention is identical except that the
different heads share a single set of keys and values. The code for (incremental) multi-query (self) attention
is identical to the code listed above for multi-head attention, except that we remove the letter "h" from the
tf.einsum equations where it represents the "heads" dimension of K, V , Pk, or Pv .
From the original paper
"""

# converted from tf and einsum to pytorch

def MultiquerySelfAttentionIncremental(x, prev_K, prev_V, P_q, P_k, P_v, P_o):
  # all queries to one key and value
  # P_q of shape [h, d, k]
  q = x @ P_q.transpose(1,2)
  k = x @ P_k 
  v = x @ P_v 
  K = torch.concat([prev_K, k.unsqueeze(2)], dim=2)
  V = torch.concat([prev_V, v.unsqueeze(2)], dim=2)

  logits = q @ K.transpose(1, 2)
  weights = torch.softmax(logits)
  o = weights @ V.transpose(1, 2)
  y = o @ P_o.transpose(1, 2)
  return y, K, V 

# basically same as self attention except using a constant k/v
def MultiquerySelfAttentionBatched(X, M, mask, P_q, P_k, P_v, P_o):
  Q = X @ P_q.transpose(1,2)
  K = M @ P_k 
  V = M @ P_v 
  logits = Q @ K.transpose(1, 2)
  weights = torch.softmax(logits + mask, dim=-1)
  O = weights @ V.transpose(1,2)
  Y = O @ P_o.transpose(1, 2)
  return Y
