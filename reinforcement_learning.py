import gym
import numpy as np
import time
import matplotlib.pyplot as plt

# AI gym is a set of environments to train reinforcement learning models
# each environment has an observation space (environment, states) and action space (actions we can take)
# setting up the FrozenLake environment
env = gym.make('FrozenLake-v1')

# env.reset() reset env to its starting state
# action = env.action_space.sample() picks random action

# takes a step and records the state we enter, reward amount, lose/win, more info
# observations = env.step(action)
# print(observations)

# number of states is 16, number of actions to take in each state is 4 (up, down, left, right)
STATES = env.observation_space.n
ACTIONS = env.action_space.n
Q = np.zeros((STATES, ACTIONS))
EPISODES = 2000 # how many times to run environment from beginning
MAX_STEPS = 100 # max num steps allowed in each run
LEARNING_RATE = 0.81
GAMMA = 0.96
epsilon = 0.9 # start w/ 0.9 chance of picking a random action

def pick_action(state):


    if np.random(0, 1) < epsilon:
        # pick random action
        action = env.action_space.sample()
    else:
        # pick max reward action
        action = np.argmax(Q[state, :])
    return action

# Q[state,action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])
rewards = []
for episode in range(EPISODES):
    state = env.reset()[0]

    for _ in range(MAX_STEPS):
        if np.random.uniform(0, 1) < epsilon:
            # pick random action
            action = env.action_space.sample()
        else:
            # pick max reward action
            action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, info = env.step(action)
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        if terminated or truncated:
            rewards.append(reward)
            epsilon -= 0.001
            break

print(Q)
print(f"Average Reward: {sum(rewards)/len(rewards)}")

# making plot of average rewards


def get_average(values):
    return sum(values)/len(values)


avg_rewards = []
for i in range(0, len(rewards), 100):
    avg_rewards.append(get_average(rewards[i:i+100]))

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100s)')
plt.show()
