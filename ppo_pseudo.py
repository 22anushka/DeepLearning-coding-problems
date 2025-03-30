# code from Umar Jamil Youtube - RLHF video
# offline policy (language model with params) -> sample trajectories based on a prompt to get multiple responses. We calculate rewards, advantages, log-probabilities etc.
# use the online model with a mini-batch of tranjectories and run gradient ascent to train the online policy
# after n epochs / mini-batches of the online_policy, we update the offline policy
# we dont sample trajectory for every step

# basically what happens in the main ppo loop
def off_policy_learning(frozen_model, model_to_train, reward_model, optimizer):
  for k in range(num_global_epochs):

    # the model acting as offline policy, get trajectories
    trajectories = sample_trajectories(model_to_train)

    # compute the rewards, log_probabilities, advantages, and KL-divergences between frozen model and model to train
    # reward model gives you the rewards
    # log_probabilities from the policy model here, the offline-policy model
    # advantage is calculated from the value function - difference between the estimated state value and the actual return (or reward signal) observed, or from the critic model
    trajectories = update_with_info(trajectories, reward_model, model_to_train, frozen_model)

    for j in range(num_ppo_epochs):
      mini_batch = get_random_mini_batch(trajectories) # these trajectories have the information for the reward, log_probabilities, advantages, etc. -> just sampling

      loss = ppo_algorithm(mini_batch)
      loss.backward()
      optimizer.step() # run gradient ascent -> model is the online policy here
  
