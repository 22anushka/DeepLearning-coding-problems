# choose the top-k values to pick from 
def top_k_sampling(logits, k=0.9):
  k = 1 if k==0 else k*len(logits)
  stable_logits = np.max(logits, axis=-1, keep_dims=True)
  probs = np.exp(logits-stable_logits)/np.sum(np.exp(logits-stable_logits))

  # sort probabilities, get indices
  # IMPORTANT: descending order sorting
  
  sorted_idx = np.argsort(probs, axis=-1)[::-1]

  # create mask for top_probs based on top k
  # technically want only first k
  top_k_mask = np.zeros_link(probs)
  top_k_mask[sorted_idx[:k]] = 1

  to_choose = top_k_mask * probs
  
  # normlize the probabilites for np.random.choice()
  norm_probs = to_choose/np.sum(to_choose)

  return np.random.choice(len(norm_probs), p=norm_probs)
