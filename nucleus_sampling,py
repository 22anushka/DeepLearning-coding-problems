# top p
import numpy as np
def p_sampling(logits, p=0.9):
  
  # convert logits into probabilities
  stable_logits = np.max(logits, axis=-1, keep_dims=True)
  probs = np.exp(logits-stable_logits)/np.sum(np.exp(logits-stable_logits))

  # sort the probabilities,
  # IMPORTANT: want decsending order
  probs_sorted = np.sort(probs, axis=-1)[::-1] # reverse order
  probs_idx = np.argsort(probs, axis=-1)[::-1]

  cum_probs = np.cumsum(probs_sorted)

  # want index of the value largest value that is <= p. argmax finds the index of the first index where condition is true (i.e. first occurance of max)
  cum_idx = np.argmax(cum_probs>=p)

  # everything after cum_idx should be zeroe-d out
  top_p_mask = np.zeros_like(probs)
  # everything mask and below can be considered to be 1
  top_p_mask[probs_idx[:cum_idx+1]]=1

  # want to apply this mask to the probabilities
  choice_probs = probs * top_p_mask
  
  # probabilities do not sum to 1 yet, they need to be normalized
  # hence
  norm_probs = choice_probs / np.sum(choice_probs, axis=-1, keepdims=True)
  print(norm_probs)

  return np.random.choice(norm_probs)
