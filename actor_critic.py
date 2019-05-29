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
        self.output_score = nn.Linear(128, 2)
        self.value_score = nn.Linear(128, 1)
        
        self.gamma = 0.99
        self.rewards = []
        self.saved_actions = []

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        output = self.output_score(x)
        output = torch.sigmoid(output)
        value = self.value_score(x)
        return output, value

policy = PolicyNetwork()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

def choose_action(state):
    x = torch.from_numpy(state).float()
    output_scores, state_value = policy(x)
    m = Categorical(output_scores)
    action = m.sample()
    policy.saved_actions.append((m.log_prob(action), state_value))
    return action.item()

def update_policy():
    R = 0
    actor_loss = []
    critic_loss = []
    disc_rewards = []
    for r in policy.rewards[::-1]:
        R = r + policy.gamma * R
        disc_rewards.insert(0, R)
    disc_rewards = torch.tensor(disc_rewards)
    disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std()\
                                                         + np.finfo(np.float32).eps.item())

    for (log_prob, value), R in zip(policy.saved_actions, disc_rewards):
        advantage = R - value.item()
        actor_loss.append(-log_prob * advantage)
        critic_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()
    actor_loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
    actor_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]

for i_episode in range(1000):
    current_state = env.reset()
    for t in range(10000):
        #env.render()
        action = choose_action(current_state)
        observation, reward, done, _ = env.step(action)
        policy.rewards.append(reward)

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break

        current_state = observation

    update_policy()

#env.close()