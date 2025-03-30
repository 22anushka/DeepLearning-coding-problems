def log_prob(logits, labels, average_log_prob:bool, label_pad_token_id: int=-100, is_encoder_decoder:bool= False):
  # want to compute the log probabilities of the given labels under the given logits
  # since we are dealing with decoder architecture, need to shift lavels and logits by 1
  labels = labels[:, 1:].clone()
  logits = logits[:, :-1, :] # everything except for the last token
  loss_mask = labels != label_pad_token_id

  # dummy tokens
  labels[labels == label_pad_token_id] == 0

  # this is important -> basically saying, take log_softmax of the logits across the last dimension, then select the log_probability corresponding to the token
  # by dim 2 and unsqueeze and squeeze 2 they are trying to refer to the seq dimension i.e. tokens
  per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

  # then, sum the log probabilites
  return (per_token_logps * loss_mask).sum(dim=-1)

def dpo_loss(model_preferred, model_dispreffered, ref_preferred, ref_dispreffered, beta=0.5):
  pref_probs = model_preferred - model_dispreffered
  ref_probs = ref_preferred - ref dispreferred

  loss = -F.logsigmoid(beta * (prefered_probs - ref_probs)).mean(dim=-1)
  return loss, pref_probs, ref_probs


# dpo does not require any reward model only requires reference dataset:
def dpo(model, ref_model, preferred_inputs, dispreferred_inputs):
  optimizer.zero_grad()
  model_prefered_log_prob = log_prob(model(prompt_prefered_ids, attention_mask=prompt_prefered_mask).logits, prompt_prefered_ids)
  model_disprefered_log_prob = log_prob(model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, prompt_disprefered_ids)

  ref_prefered_log_prob = log_prob(ref_model(prompt_prefered_ids, attention_mask=prompt_prefered_mask).logits, prompt_prefered_ids)
  ref_disprefered_log_prob = log_prob(ref_model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, prompt_disprefered_ids)

  loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(model_prefered_log_prob, model_disprefered_log_prob,
                                ref_prefered_log_prob, ref_disprefered_log_prob,
                                beta=beta)

  loss.backward()
  optimizer.step()
