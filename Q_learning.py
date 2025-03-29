import torch
import numpy as np


# Q-Learning
# with help from https://www.geeksforgeeks.org/q-learning-in-python/
def q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay=0.99, epsilon_min=0.01):
    """
    Q-Learning algorithm implementation.
    
    Args:
        env: Environment with discrete action and state spaces (state, action)
        num_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        epsilon_decay: Rate at which epsilon decreases after each episode
        epsilon_min: Minimum value of epsilon
        
    Returns:
        Q: Learned Q-table
    """
    # Initialize Q-table
    """
    Q-values represent the expected rewards for taking an action in a specific state. 
    Here, these values are updated over time using the Temporal Difference (TD) update rule.
    Q(s, a) = Q(s, a) + alpha*(R+ gamma*Q(s', a') - Q(s, a)) 
    Can also be updated using Bellman equation Q(s,a) = R(s,a ) + gamma argmaxQ(s', a)
    """
    print(env.__dir__())

    # TODO: Initialize Q-table with zeros for all state-action pairs
    # Q-table holds the rewards
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q_table = torch.zeros((n_states, n_actions))
    
    for episode in range(num_episodes):
        # Reset environment for new episode
        # TODO: Reset environment and get initial state
        # method at the start of each episode, ensuring the agent begins from a fresh, initial state. 
        
        state = env.reset()
        
        done = False
        while not done:
            # Select action using epsilon-greedy policy
            # The ϵ-greedy policy helps the agent decide which action to take based on the current Q-value estimates
            # TODO: Implement epsilon-greedy action selection
            # try to balance exploitation - pick with a probability (1-eps) and exploration - pick with a probability (eps)
            if torch.torch.rand(1).item() < epsilon:
              # choose exploration
              action = torch.randint(0, n_actions, (1,)).item()
            else:
              # best action
              action = torch.argmax(Q_table[state])
            
            # Take action and observe reward and next state
            # TODO: Execute action and get next_state, reward, done status
            """
            # env has the information
            next_state = (state + 1) % n_states
            done = True if next_state == goal_state else False # goal_state is the last state
            reward = 1 if done else 0 # sparse reward system since no reward has been defined
            """
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-value for the state-action pair
            # TODO: Implement Q-learning update rule: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s')) - Q(s,a))
            Q_table[state, action] += alpha*(reward + gamma*torch.max(Q_table[next_state]) - Q_table[state, action])
            
            # Update current state to next state
            # TODO: Update state
            state = next_state
            
        # Update exploration rate
        # TODO: Decay epsilon after each episode
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    return Q_table

  
