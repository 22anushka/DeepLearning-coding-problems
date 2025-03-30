# advantage estimation for language based rlhf
# generalized advantage estimation

def calc_adv():
  lastgaelam = 0 # stores the gae value from the previous step in the loop
  advantages_reversed = [] # so we are calculating the advantage in reverse since we are going from the last token to the first token
  gen_len = rewards.shape[-1] # length of the episode? -> how many steps you took to get to this generation?

  # masking values and rewards
  # here, it masks the prompt tokens and pad tokens basically
  values = values * mask
  rewards = rewards * mask

  # whitening rewards -> normalizes rewards across episodes for stability?

  # gen_len is the length of the generated sample / trajectory
  # going from timestep T=0 to 0
  for t in reversed(range(gen_len)):
    next_values = values[:, t+1] if t < gen_len - 1 else 0.0  # Value function evaluated at time (t+1) since we are interested of the "generated" token; next state value
    # id t == gen_len-1 then it's the last state, no next state

    # From the formula of GAE: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) (V(s_t) has to be subtracted since it is "advantage")
    # TD errror
    delta = rewards[:, t] + gamma * next_value - values[:, t]

    lastgaelam = delta + gamma * lam * lastgaelam # lam helps control tradeoff between bias and variance (stop too early vs. keep generating) weighted avg recursively to balance

    # stores advantages
    advantages_reversed.append(lastgaelam)

  advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

  returns = advantages + values # since we are actually trying to return Q-value and advantage is Q(s, a) - V(s)

  advantages = masked_whiten(advantages, mask) # normalizes advantage while ignoring the masked elements -> better for training stability zero mean and unit variance


def calc_values():
  # a trainable value head (often a small linear layer) is added on top of the model to predict state values.
  # The ValueHead helps in training the critic by minimizing the value loss (e.g., MSE loss between predicted values and actual returns).
  logits, _, values = model(**input_kwargs)
