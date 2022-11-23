import numpy as np
import random
from collections import namedtuple, deque
# Importing the model
from dqn import DQN

import torch
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)  # 使用队列保存样本
        self.batch_size = batch_size  # 一批训练数据的大小
        self.experiences = namedtuple(  # 使用namedtuple来记录每条样本的信息
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)  # 设置随机数种子

    def add(self, state, action, reward, next_state, done):
        # 向经验池中保存一条样本
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # 随机采样出一批样本
        experiences = random.sample(self.memory, k=self.batch_size)
        # 将采样样本转换为torch.tensor类型，并且传送到计算设备上
        states = torch.cat([e.state for e in experiences if e is not None]).float()
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(DEVICE)
        next_states = torch.cat(
            [e.next_state for e in experiences if e is not None]).float()
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences
                       if e is not None]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQNAgent():
    def __init__(self,
                 action_size,
                 seed,
                 lr=1e-3,
                 gamma=0.99,
                 buffer_size=int(1e5),
                 batch_size=64,
                 update_every=100,
                 target_update_every=200):

        self.action_size = action_size
        random.seed(seed)
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update_every = target_update_every
        self.gamma = gamma

        # Q- Network
        self.qnetwork_local = DQN(action_size, seed, DEVICE).to(DEVICE)
        self.qnetwork_target = DQN(action_size, seed, DEVICE).to(DEVICE)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # 经验回放池
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # 步数记录，用于策略更新
        self.t_step = 0
        self.target_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)  # 向经验池保存样本

        self.t_step = (self.t_step + 1) % self.update_every
        self.target_step = (self.target_step + 1) % self.target_update_every
        if self.t_step == 0:  # 每隔一定步数训练一次评估网络
            if len(self.memory) > 10000:  # 当回访池容量大于10000时开始采样并训练
                experience = self.memory.sample()
                self.learn(experience)
        if self.target_update_every == 0:  # 每隔一定步数更新一次策略网络
            self.hard_update()

    def act(self, state, eps=0):
        state = state.float()  # 将数据转换为浮点型

        # 使用网络的eval模式，前向传播时不自动计算网络梯度
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        # 恢复train()模式，允许计算梯度并进行网络的更新
        self.qnetwork_local.train()

        # Epsilon-greedy动作选择
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # 学习评估网络参数
        criterion = torch.nn.MSELoss()  # Loss使用均方误差
        self.qnetwork_local.train()
        self.qnetwork_target.eval()

        predicted_targets = self.qnetwork_local(states).gather(1, actions)  # 真实Q值
        with torch.no_grad():  # 预测Q值
            labels_next = self.qnetwork_target(next_states).detach().max(
                1)[0].unsqueeze(1)
        labels = rewards + (self.gamma * labels_next * (1 - dones))

        # 计算误差与梯度更新
        loss = criterion(predicted_targets, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def hard_update(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
