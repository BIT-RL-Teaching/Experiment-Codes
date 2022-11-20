# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from pylab import mpl, plt

# 1. 超参数定义
BATCH_SIZE = 32     # 一个batch内样本数量
LR = 0.01           # 学习率
EPSILON = 0.9       # 贪心策略epsilon
GAMMA = 0.9         # 折扣率
TARGET_REPLACE_FREQ = 100   # 目标Q网络更新频率
MEMORY_CAPACITY = 2000      # 经验回放池容量

env = gym.make("CartPole-v0")               # 创建CartPole-v0环境
env = env.unwrapped
N_ACTIONS = env.action_space.n              # 动作个数：2
N_STATES = env.observation_space.shape[0]   # 状态个数：4

# 2. DQN网络结构
class Net(nn.Module):
    def __init__(self):
        # 两层全连接网络
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)      # 第一个全连接层
        self.fc1.weight.data.normal_(0, 0.1)    # 初始化权重（均值0，方差0.1的正态分布）
        self.out = nn.Linear(50, N_ACTIONS)     # 第二个全连接层
        self.out.weight.data.normal_(0, 0.1)    # 初始化权重

    def forward(self, x):
        x = self.fc1(x)                 # 从输入层到隐藏层
        x = F.relu(x)                   # relu激励函数
        actions_value = self.out(x)     # 从隐藏层到输出层
        return actions_value            # 返回动作值
        
# 3. DQN算法
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # 使用Net类创建评估与目标网络
        self.learn_step_counter = 0     # 记录训练步数
        self.memory_counter = 0         # 记录经验池容量
        # 初始化经验池空间，每条经验为一个四元组（s,a,r,s_）
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        # 使用Adam优化器与均方误差损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        
    def choose_action(self, x):
        # 动作选择函数
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 在x的第0维度上增加1个维度
        # 使用epsilon-greedy策略选择动作
        if np.random.uniform() < EPSILON:  # 生成[0,1)随机数，如果小于eps就选取最优动作
            # 在当前状态下，利用评估网络计算动作值
            actions_value = self.eval_net.forward(x)
            # 选择具有最大Q值的动作索引
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]  # 返回最优动作
        else:  # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        # 向经验池中存储记录
        transition = np.hstack((s, [a, r], s_))  # 在水平方向上拼接数组
        # 如果经验池满了，覆盖原有的记录
        index = self.memory_counter % MEMORY_CAPACITY  # 计算存储位置
        self.memory[index, :] = transition  # 置入记录
        self.memory_counter += 1            # 计数器+1
        
    
    def learn(self):
        # 策略更新函数
        # 每隔一定步数更新目标网络
        if self.learn_step_counter % TARGET_REPLACE_FREQ == 0:
            # 将评估网络的参数赋给目标网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 从经验池中采样batch数据
        # 随机抽取BATCH_SIZE个数值，作为batch数据的索引
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # 抽取索引对应的transition，存入b_memory
        b_memory = self.memory[sample_index, :]
        # 从b_memory中提取出对应的s,a,r,s_并分别存储
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 需要将动作转换为64-bit int数据类型
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 获取每个transition的目标值与评估值，并利用损失函数和优化器更新评估网络参数
        # 计算出每个可能动作对应的Q值，然后对每行对应索引b_a的Q值提取进行聚合
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # 计算下一状态对应的动作Q值，并且不反向传递误差
        q_next = self.target_net(b_s_).detach()
        # 选择每一行的最大Q值，并且使用view()将张量转换为(BATCH_SIZE,1)的形状
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # 计算评估值与目标值之间的误差
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        self.optimizer.step()  # 更新网络参数


dqn = DQN()  # 初始化DQN类
rewards = []  # 记录每个episode的奖励
ma_rewards = []  # 记录奖励的滑动平均值
print("\nCollecting experience...")
for i_episode in range(800):  # 执行800个episode
    s = env.reset()  # 重置环境
    ep_r = 0  # 初始化累计奖励
    while True:  # 开始一个episode
        # env.render()  # 显示游戏动画
        a = dqn.choose_action(s)  # 输入对应状态s，输出选择的动作a
        s_, r, done, info = env.step(a)  # 执行动作，获取反馈

        # 修改奖励值，目的是加速训练（也可以不修改）
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2

        dqn.store_transition(s, a, new_r, s_)  # 存储样本
        ep_r += new_r  # 累积episode获得的奖励

        # 如果累计的transition达到了经验池的最大容量，开始更新评估网络参数
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
        # 如果游戏达到结束条件，记录统计信息，结束episode
        if done:
            print('Ep: %s | Ep_r: %s' % (i_episode, round(ep_r, 2)))
            rewards.append(ep_r)
            break
        s = s_  # 更新状态
    # 计算episode的滑动平均奖励值
    if ma_rewards:
        ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_r)
    else:
        ma_rewards.append(ep_r)

# 画图
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(rewards, label='总奖励')
plt.plot(ma_rewards, label='平均奖励')
plt.legend(loc=0)
plt.show()
