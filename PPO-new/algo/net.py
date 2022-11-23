import os
import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
import math


class Mujoco_Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Mujoco_Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, 150)
        self.l2 = nn.Linear(150, 150)
        self.alpha_head = nn.Linear(150, act_dim)
        self.beta_head = nn.Linear(150, act_dim)

    def forward(self, obs):
        a = torch.tanh(self.l1(obs))
        a = torch.tanh(self.l2(a))
        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0
        return alpha, beta

    def get_dist(self, obs):
        alpha, beta = self.forward(obs)
        dist = Beta(alpha, beta)
        return dist

    def evaluate_mode(self, obs):
        alpha, beta = self.forward(obs)
        mode = alpha / (alpha + beta)
        return mode


class Mujoco_Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Mujoco_Critic, self).__init__()
        self.C1 = nn.Linear(obs_dim, 150)
        self.C2 = nn.Linear(150, 150)
        self.C3 = nn.Linear(150, 1)

    def forward(self, obs):
        v = torch.tanh(self.C1(obs))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v

# 该函数用于对网络参数进行初始化
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Atari_Actor_Critic(nn.Module):
    def __init__(self, channel, act_dim, hidden_size=512):
        '''
        channel: 图片输入的通道数。借鉴DQN论文中的做法将相邻的四帧堆叠作为网络的输入，
        因此默认channel=4。需注意，这里通道的意义不是图像处理中的RGB颜色通道，而是每个通道代表
        一帧灰度图像
        act_dim: 智能体输出的动作维度。BattleZone中可选18个离散动作，因此act_dim=18。
        '''
        super(Atari_Actor_Critic, self).__init__()

        init_for_main = lambda m: init(m,
           # 对网络中weight进行正交初始化，这是常用的让训练更稳定的技巧
           nn.init.orthogonal_,
           # 把网络中的bias初始化为0
           lambda x: nn.init.constant_(x, 0),
           # gain是参数初始化时的缩放因子，例如在正交初始化中，使用gain=1与使用gain=0.1相比，前者初始化的网络参数的标准差是后者的10倍
           # 由于非线性激活函数将影响标准差，训练时可能会遇到梯度消失等问题，
           # 因此不同的非线性激活函数有不同的推荐使用的gain,使用推荐的gain能使得训练更稳定。
           # 详情可参见https://pytorch.org/docs/stable/nn.init.html
           nn.init.calculate_gain('relu'))  # relu推荐使用的gain是sqrt(2)

        self.main = nn.Sequential(
            # nn.Conv2d的四个参数的含义分别为输入通道数（也即frame stack的数量）、输出通道数、卷积核大小、步长
            init_for_main(nn.Conv2d(channel, 32, 8, stride=4)), nn.ReLU(),
            init_for_main(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_for_main(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            # nn.Linear的两个参数的含义分别为全连接层的输入维度和输出维度
            # 由于Atari游戏经过预处理的图片输入的宽和高都是84，这里可以硬编码指定全连接层的输入维度
            init_for_main(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU()
        )

        init_for_actorhead = lambda m: init(m,
                                            nn.init.orthogonal_,
                                            lambda x: nn.init.constant_(x, 0),
                                            gain=0.01)  # 针对softmax激活函数的gain使用0.01
        # actor_head的输出维度是act_dim，输出的含义是在当前状态下各动作的概率分布
        self.actor_head = init_for_actorhead(nn.Linear(hidden_size, act_dim))

        init_for_critichead = lambda m: init(m,
                                             nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0))
        # critic_head的输出维度是1，输出的含义是对当前状态s的价值估计，即V(s)
        self.critic_head = init_for_critichead(nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        '''
        inputs: 一批待送入网络的状态
        若在与环境交互时调用，维度为(1, 4, 84, 84)，其中4是frame stack的数量
        若在训练时调用，维度为(batchsize, 4, 84, 84)
        '''
        # 图片输入中各像素的取值范围为[0, 255]，归一化后送入共享的卷积层
        x = self.main(inputs / 255.0)
        # 返回critic_head输出的价值估计和actor_head输出的动作概率分布
        return self.critic_head(x), torch.softmax(self.actor_head(x), dim=-1)


class CarRacing_Actor_Critic(nn.Module):
    def __init__(self, channel, act_dim, hidden_size=512):
        super(CarRacing_Actor_Critic, self).__init__()

        init_for_main = lambda m: init(m,
                                       nn.init.orthogonal_,
                                       lambda x: nn.init.constant_(x, 0),
                                       nn.init.calculate_gain('relu'))
        self.main = nn.Sequential(
            # channel的物理意义是frame stack的数量
            init_for_main(nn.Conv2d(channel, 32, 8, stride=4)), nn.ReLU(),
            init_for_main(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_for_main(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            # 硬编码到适配CarRacing的维度
            init_for_main(nn.Linear(2048, hidden_size)), nn.ReLU()
        )

        init_for_actorhead = lambda m: init(m,
                                            nn.init.orthogonal_,
                                            lambda x: nn.init.constant_(x, 0),
                                            )  # TODO for softplus, what is the gain? don't know so delete
        # for CarRacing, act_dim=3
        self.alpha_head = init_for_actorhead(nn.Linear(hidden_size, act_dim))
        self.beta_head = init_for_actorhead(nn.Linear(hidden_size, act_dim))

        init_for_critichead = lambda m: init(m,
                                             nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0))
        self.critic_head = init_for_critichead(nn.Linear(hidden_size, 1))

    def forward(self, obs):
        # check: obs.shape should be (batch, 4, 96, 96)
        x = self.main(obs / 255.0)
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0
        return self.critic_head(x), alpha, beta

    def get_dist(self, obs):
        value, alpha, beta = self.forward(obs)
        dist = Beta(alpha, beta)
        return value, dist

    def mode(self, obs):
        value, alpha, beta = self.forward(obs)
        mode = alpha / (alpha + beta)
        return value, mode






















