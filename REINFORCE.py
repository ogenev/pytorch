import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from torch.distributions import Categorical

env = gym.make('CartPole-v0')

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(4, 128)
        self.linear2 = nn.Linear(128, 2)
        
        self.gamma = 0.99
        self.rewards = []
        self.log_probs = []

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        output = self.linear2(x)
        output = torch.sigmoid(output)
        return output

policy = PolicyNetwork()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

def choose_action(state):
    x = torch.from_numpy(state).float()
    output_scores = policy(x)
    m = Categorical(output_scores)
    action = m.sample()
    policy.log_probs.append(m.log_prob(action))
    return action.item()

def update_policy():
    R = 0
    policy_loss = []
    disc_rewards = []
    for r in policy.rewards[::-1]:
        R = r + policy.gamma * R
        disc_rewards.insert(0, R)
    disc_rewards = torch.tensor(disc_rewards)
    disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std()\
                                                         + np.finfo(np.float32).eps.item())
    #disc_rewards = disc_rewards - disc_rewards.mean()
    for log_prob, R in zip(policy.log_probs, disc_rewards):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()

    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.log_probs[:]

for i_episode in range(1000):
    current_state = env.reset()
    for t in range(10000):
        env.render()
        action = choose_action(current_state)
        observation, reward, done, _ = env.step(action)
        policy.rewards.append(reward)

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break

        current_state = observation

    update_policy()

env.close()