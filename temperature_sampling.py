# basically, scale the softmax exponent calculation probability by temperature to scale probability values towards one hot if temperature is low and towards uniform if temperature is high
def temp_sampling(logits, t=0.6):

  # convert logits to probs while scaling by temperature
  stable_logits = np.max(logits, axis=-1, keep_dims=True)
  t = 0.1 if t == 0 else t
  temp_scaled = np.exp((logits-stable_logits)/t)/np.sum((logits-stable_logits)/t)

  # can call sampling method given the probabilities from here
  # random sampling
  return np.random.choice(len(temp_scaled), p=temp_scaled)
