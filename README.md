# MONTE CARLO CONTROL ALGORITHM

## AIM
The aim of Monte Carlo Control is to develop an optimal policy for a Markov Decision Process (MDP) by estimating the value function and improving the policy through repeated sampling and evaluation of episodes. This technique is used in reinforcement learning to find the best actions to take in each state in order to maximize cumulative rewards..

## PROBLEM STATEMENT
Monte Carlo Control is a reinforcement learning method, to figure out the best actions for different situations in an environment. The provided code is meant to do this, but it's currently having issues with variables and functions.

## MONTE CARLO CONTROL ALGORITHM
# Step 1:
Initialize Q-values, state-value function, and the policy.

# Step 2:
Interact with the environment to collect episodes using the current policy.

# Step 3:
For each time step within episodes, calculate returns (cumulative rewards) and update Q-values.

# Step 4:
Update the policy based on the improved Q-values.

# Step 5:
Repeat steps 2-4 for a specified number of episodes or until convergence.

# Step 6:
Return the optimal Q-values, state-value function, and policy.

## MONTE CARLO CONTROL FUNCTION
```
import numpy as np
from collections import defaultdict

def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    nS, nA = env.observation_space.n, env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    V = defaultdict(float)
    pi = defaultdict(lambda: np.random.choice(nA))  
    Q_track = []
    pi_track = []
    for episode in range(n_episodes):
        epsilon = max(init_epsilon * (epsilon_decay_ratio ** episode), min_epsilon)
        alpha = max(init_alpha * (alpha_decay_ratio ** episode), min_alpha)  
        trajectory = generate_trajectory(select_action, Q, epsilon, env, max_steps)
        n = len(trajectory)
        G = 0  
        for t in range(n - 1, -1, -1):
            state, action, reward, _, _ = trajectory[t]
            G = gamma * G + reward
            if first_visit and (state, action) not in [(s, a) for s, a, _, _, _ in trajectory[:t]]:
                Q[state][action] += alpha * (G - Q[state][action])
                V[state] = np.max(Q[state])
                pi[state] = np.argmax(Q[state])
        Q_track.append(Q.copy())
        pi_track.append(pi.copy)
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
![image](https://github.com/Saibandhavi75/monte-carlo-control/assets/94208895/ec6c5445-1af7-4d78-ac2b-b79f9f0ca49b)


## RESULT:

Monte Carlo Control successfully learned an optimal policy for the specified environment.
