# need to have a k_cache and a v_cache to store the kv
# in the init
"""
b_max = max_batch_size, l_max = max_seq_len, n = num_heads, d_h = d_model//n
self.cache_k = torch.zeros((b_max, l_max, n, d_h))
self.cache_v = torch.zeros((b_max, l_max, n, d_h))

# in grouped query attention, the n would be n_kv i.e. number of heads for the kv, but it's the same otherwise
"""
def repeat_kv(kv, num_rep):
  # just unsqueeze and repeat
  if num_rep == 1:
    return kv
  else:
    b, l, n_kv, d_hkv = kv.shape
    return kv.unsqueeze(3).expand(b, l, n_kv, num_rep, d_hkv).reshape(b, l, n_kv*num_rep, d_hkv)


def GroupedQueryAttention(x, start_pos, w_Q, w_K, w_V, num_heads, n_kv):
  # ideally the other params will be stored inside init as self. variables
  # start position is to understand
  n = num_heads
  b, l, d_model = x.shape
  Q = w_Q(x)
  K = w_K(x)
  V = w_V(x)
  d_h = d_model // n
  d_hkv = d_h # head dimension should still be the same since we will just repeat the kv for the queries


  # view it as (b, l, n, d_h)
  Q = Q.view(b, l, n, d_h)
  K = K.view(b, l, n_kv, d_hkv)
  V = V.view(b, l, n_kv, d_hkv)

  # replace the entry in the cache with the calculated k, v for every batch upto seq len
  # in kv_caching l = 1
  self.cache_k[:b, start_pos:start_pos+l] = K
  self.cache_v[:b, start_pos:start_pos+l] = V

  # retrieve everything cached before this particular token / keys and values so far
  keys = self.cache_k[:b, 0:start_pos+l]
  values = self.cache_k[:b, 0:start_pos+l]


  # Repeat the heads of KV to reach the number of heads of the query (not the most efficient but the "expand and concat" has been proposed in the original code)
  keys = repeat_kv(keys, n//n_kv)
  values = repeat_kv(values, n//n_kv)

  Q = Q.transpose(1, 2)
  keys = keys.transpose(1, 2)
  values = values.transpose(1, 2)

  # transpose 2, 3 since keys is currently of shape b, l, n, d_h need to be b, l, d_h, n for correct multiplication
  # resulting in bln
  scores = Q @ keys.transpose(2,3)
  # softmax, value, concatenation and projection
