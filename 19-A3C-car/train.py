from torch.autograd import Variable
from collections import deque
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import torch

from ACmodel import ActorCritic
from envs import create_env
import constants as c


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(c.seed + rank)

    # 创建仿真器
    env = create_env(c.env_name)
    env.seed(c.seed + rank)
    # 创建AC模型
    model = ActorCritic(env.observation_space.shape[0], c.output_size)
    # 创建优化器
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=c.lr)

    # 将模型转到训练的模式
    model.train()

    # 准备仿真器
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0

    # 训练
    while True:
        # 同步各个线程的模型 (state_dict方法返回整个环境的状态)
        model.load_state_dict(shared_model.state_dict())

        # 如果游戏结束则reset cx, hx
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        # 准备 values, log_pi, rewards, entropies loggers
        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(c.num_steps):
            episode_length += 1

            value, logit, (hx, cx) = model(( Variable(state.unsqueeze(0)), (hx, cx) ))	

            prob = F.softmax(logit, dim = 1)
            log_prob = F.log_softmax(logit, dim = 1)

            entropy = - (log_prob * prob).sum(1, keepdim = True)
            entropies.append(entropy)

            # 根据分布概率抽样行为
            action = prob.multinomial(num_samples = 1).data     # 一个 tensor
            log_prob = log_prob.gather(1, Variable(action))     # 选择 log_prob[action]

            # 仿真器行为
            state, reward, done, _ = env.step(action.numpy())    # 转tensor(rank=1) 为 float

            # 结束游戏如果浪费太长时间
            done = done or episode_length >= c.max_episode_length
            # 奖励的标准化
            reward = max(min(reward, 1), -1)
            # 线程锁
            with lock:
                counter.value += 1
            if done:
                episode_length = 0
                # -------------------------------------------------------------------------------#
                # -------------------------------------------------------------------------------#
                # ---------------------             2 填空           -----------------------------#
                # -------------------------------------------------------------------------------#
                # -------------------------------------------------------------------------------#
                #2 如果游戏结束则重置仿真器
                state = env.reset()

            # 用新的 state, logging values, log_pi, r来代替state
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))  
        policy_loss = 0
        value_loss = 0

        R = Variable(R)
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = c.gamma * R + rewards[i]
            advantage = R - values[i]

            value_loss = value_loss + 0.5 * advantage.pow(2)

            # 生成 Advantage Estimataion
            delta_t = rewards[i] + c.gamma * values[i + 1].data - values[i].data
            gae = gae * c.gamma * c.tau + delta_t

            # 梯度下降的Loss_pi, minus
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - c.entropy_coef * entropies[i]

        #3 梯度的后馈传递
        optimizer.zero_grad()
        (policy_loss + c.value_loss_coef * value_loss).backward()
        # 修剪梯度以防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
        ensure_shared_grads(model, shared_model)

        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             3 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        #3 梯度被如backward()函数计算好后，调用step()函数进行参数更新
        optimizer.step()

