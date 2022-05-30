import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import numpy as np
from collections import namedtuple
import math

GAMMA = 0.9
lr = 0.1
EPSION = 0.9
buffer_size = 10000
batch_size = 32
num_episode = 100000
target_update = 10
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
tau = 0.02

#定义神经网络
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear1.weight.data.normal_(0, 0.1)
        # self.Linear2 = nn.Linear(hidden_size, hidden_size)
        # self.Linear2.weight.data.normal_(0, 0.1)
        self.Linear3 = nn.Linear(hidden_size, output_size)
        self.Linear3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # print('x: ', x)
        x = F.relu(self.Linear1(x))
        # x = F.relu(self.Linear2(x))
        x = torch.sigmoid(self.Linear3(x))
        return x

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear1.weight.data.normal_(0, 0.1)
        # self.Linear2 = nn.Linear(hidden_size, hidden_size)
        # self.Linear2.weight.data.normal_(0, 0.1)
        self.Linear3 = nn.Linear(hidden_size, output_size)
        self.Linear3.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        # print('s1: ', s)
        # print('a1: ', a)
        x = torch.cat([s, a], dim=1)
        # print('x: ', x)
        x = F.relu(self.Linear1(x))
        # x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x

#nametuple容器
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):#采样
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDPG(object):
    def __init__(self, input_size, action_shape, hidden_size, output_size):
        self.actor = Actor(input_size, hidden_size, action_shape)
        self.actor_target = Actor(input_size, hidden_size, action_shape)
        self.critic = Critic(input_size + action_shape, hidden_size, action_shape)
        self.critic_target = Critic(input_size + action_shape, hidden_size, action_shape)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.01)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.01)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.buffer = ReplayMemory(buffer_size)
        self.loss_func = nn.MSELoss()
        self.steps_done = 0

    def put(self, s0, a0, r, s1, done):
        self.buffer.push(s0, a0, r, s1, done)

    def select_action(self, state):
        state = torch.Tensor(state)
        a = self.actor(state)
        return a

    def update_parameters(self):
        if self.buffer.__len__() < batch_size:
            return
        samples = self.buffer.sample(batch_size)
        batch = Transition(*zip(*samples))
        # print(batch.action)
        #将tuple转化为numpy
        # tmp = np.vstack(batch.action)
        # print(tmp)
        #转化成Tensor
        state_batch = torch.Tensor(batch.state)
        action_batch = torch.Tensor(batch.action).unsqueeze(0).view(-1, 1)
        reward_batch = torch.Tensor(batch.reward)
        next_state_batch = torch.Tensor(batch.next_state)
        done_batch = torch.Tensor(batch.done)
        #critic更新
        next_action_batch = self.actor_target(next_state_batch).unsqueeze(0).detach().view(-1, 1)
        # print('batch: ', next_action_batch)

        r_eval = self.critic(state_batch, action_batch)
        # print('s: ', next_state_batch)
        # print('a: ', next_action_batch)
        r_target = reward_batch + GAMMA * self.critic_target(next_state_batch, next_action_batch).detach().view(1, -1) * done_batch
        r_eval = torch.squeeze(r_eval)
        r_target = torch.squeeze(r_target)
        loss = self.loss_func(r_eval, r_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        #actor更新
        a = self.actor(state_batch).unsqueeze(0).view(-1, 1)
        # print('a: ', a)
        loss = -torch.mean(self.critic(state_batch, a))
        self.actor_optim.zero_grad()
        loss.backward()
        # print('a: ', a)
        self.actor_optim.step()
        #soft update
        def soft_update(net_target, net):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        soft_update(self.actor_target, self.actor)
        soft_update(self.critic_target, self.critic)



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    Agent = DDPG(env.observation_space.shape[0], 1, 16, env.action_space.n)
    average_reward = 0
    for i_episode in range(num_episode):
        s0 = env.reset()
        tot_reward = 0
        tot_time = 0
        while True:
            env.render()
            a0 = Agent.select_action(s0)
            s1, r, done, _ = env.step(round(a0.detach().numpy()[0]))
            tot_time += r
            tot_reward += r
            Agent.put(s0, a0, r, s1, 1 - done) #结束状态很重要，不然会很难学习。
            s0 = s1
            Agent.update_parameters()
            if done:
                average_reward = average_reward + 1 / (i_episode + 1) * (tot_time - average_reward)
                # if i_episode % 100 == 0:
                print('Episode ', i_episode, 'tot_time: ', tot_time, ' tot_reward: ', tot_reward, ' average_reward: ', average_reward)
                break
        # if i_episode % target_update == 0:
        #     Agent.target_net.load_state_dict(Agent.net.state_dict())