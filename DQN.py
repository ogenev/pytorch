import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from collections import deque
from torch.distributions import Categorical
from random import sample
import math

env = gym.make('CartPole-v0')
batch_size = 32
gamma = 0.95
eps_start = 1.0
eps_end = 0.01
eps_decay = 200
target_update = 10
steps_done = 0

class Replay_Memory(object):
    def __init__(self, batch):
        self.batch = batch
        self.memory = deque(maxlen=1000000)
      
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, torch.FloatTensor([reward]), torch.FloatTensor([next_state]), done))

    def random_sample(self):
        return sample(self.memory, self.batch)

class Deep_Q_Network(nn.Module):
    def __init__(self):
        super(Deep_Q_Network, self).__init__()
        
        self.linear1 = nn.Linear(4, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        output = self.linear2(x)
        return output


def update_network():
    if len(replay_memory.memory) < batch_size:
        return
    
    sample = replay_memory.random_sample()

    for state, action, reward, next_state, terminal in sample:

        current_q = policy_net(state).gather(1, action.unsqueeze(1))
        expected_q = reward
        max_next_q = target_net(next_state).detach().max(1)[0]

        if not terminal:
            expected_q = reward + (gamma * max_next_q)

        loss = F.smooth_l1_loss(current_q.squeeze(), expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def choose_action(state):
    global steps_done
    random_action = np.random.rand()

    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1

    if random_action < eps_threshold:
        action = env.action_space.sample()
        return action
    else:
        output_scores = policy_net(state)
        action = output_scores.argmax()
        return action.item()
        

replay_memory = Replay_Memory(batch_size)
policy_net = Deep_Q_Network()
target_net = Deep_Q_Network()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)


for i_episode in range(1000):
    current_state = env.reset()
    for t in range(10000):
        env.render()
        current_state = torch.FloatTensor([current_state])
        action = choose_action(current_state)
        observation, reward, done, _ = env.step(action)
        reward = reward if not done else -reward
        action = torch.LongTensor([action])
        replay_memory.add_memory(current_state, action, reward, observation, done)

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break

        current_state = observation

    update_network()

    if i_episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()
