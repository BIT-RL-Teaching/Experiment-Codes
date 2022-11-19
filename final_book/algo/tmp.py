import time
import copy
import os
import shutil
# from algo.utils import normalize

import gym.spaces
import numpy as np
import torch
from torch.distributions import Categorical
import math
from algo.net import Mujoco_Actor, Mujoco_Critic, Atari_Actor_Critic, CarRacing_Actor_Critic


class PPO_Atari():
    def __init__(self,
                 output_dir,
                 device,
                 writer,
                 T_horizon,
                 n_rollout_threads,
                 obs_space,
                 act_space,
                 gamma=0.99,
                 lambd=0.95,
                 clip_rate=0.2,
                 K_epochs=10,
                 a_lr=3e-4,
                 c_lr=3e-4,  # mute
                 l2_reg=1e-3,
                 optim_batch_size=64,
                 entropy_coef=0,  # 0.001
                 entropy_coef_decay=0.9998,
                 eps=1e-5,
                 critic_loss_coef=0.5,
                 max_grad_norm=0.5,
                 ):

        self.output_dir = output_dir
        self.device = device
        self.writer = writer
        self.T_horizon = T_horizon
        self.n_rollout_threads = n_rollout_threads
        self.obs_space = obs_space
        self.act_space = act_space
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.l2_reg = l2_reg
        self.critic_loss_coef = critic_loss_coef
        self.optim_batch_size = optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.eps = eps
        self.max_grad_norm = max_grad_norm

        self.total_num_steps = None

        # for atari
        assert len(self.obs_space.shape) == 3 and isinstance(self.act_space, gym.spaces.Discrete)
        self.actor_critic = Atari_Actor_Critic(self.obs_space.shape[0], self.act_space.n).to(self.device)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.a_lr, eps=self.eps)
        self.train_steps = 0
        self.step = 0  # 仅在_rollout_insert中被读+写
        self._rollout_init()

    def _rollout_init(self):
        self.obs = torch.zeros(self.T_horizon+1, self.n_rollout_threads, *self.obs_space.shape).to(self.device)
        self.rewards = torch.zeros(self.T_horizon, self.n_rollout_threads, 1).to(self.device)
        self.value_preds = torch.zeros(self.T_horizon + 1, self.n_rollout_threads, 1).to(self.device)
        # dones[0]是无意义的，查看insert就能明白
        self.dones = torch.zeros(self.T_horizon + 1, self.n_rollout_threads, 1).to(self.device)
        # return将会在compute_returns计算后被填充
        self.returns = torch.zeros(self.T_horizon + 1, self.n_rollout_threads, 1).to(self.device)
        self.actions = torch.zeros(self.T_horizon, self.n_rollout_threads, 1).to(self.device)
        self.logprob_a = torch.zeros(self.T_horizon, self.n_rollout_threads, 1).to(self.device)


    def rollout_insert(self, obs, a, logprob_a, value_pred, r, done):
        self.obs[self.step+1].copy_(obs)
        self.actions[self.step].copy_(a)
        self.logprob_a[self.step].copy_(logprob_a)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(r)
        self.dones[self.step+1].copy_(
            torch.FloatTensor([done])  # 硬编码threads=1
        )
        self.step = (self.step + 1) % self.T_horizon  # 写self.step

    def compute_return(self):
        with torch.no_grad():
            next_value = self.actor_critic(self.obs[-1])[0]  # [0] 是在网络输出的元组中取第一个元素
        self.value_preds[-1] = next_value

        gae = 0
        for step in reversed(range(self.rewards.size(0))):  # r.size == T_horizon
            # delta表达式中使用done[step+1]而不是done[step]的原因，查看insert就能明白
            delta = self.rewards[step] \
                    + self.gamma * self.value_preds[step+1] * (1 - self.dones[step + 1]) \
                    - self.value_preds[step]
            gae = delta + gae * self.gamma * self.lambd * (1 - self.dones[step + 1])
            self.returns[step] = gae + self.value_preds[step]

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.dones[0].copy_(self.dones[-1])

    def save(self, timestep, is_evaluate, is_newbest=False):
        save_dir = os.path.join(self.output_dir, 'model')
        save_dir += '/eval' if is_evaluate else '/train'
        if is_newbest: save_dir += '/best_model'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        postfix = 'best' if is_newbest else f'ts{timestep}'

        torch.save(self.actor_critic.state_dict(), save_dir + f"/actor_critic_{postfix}.pth")

    def load(self, load_dir):
        self.actor_critic.load_state_dict(torch.load(load_dir))

    def select_action_for_vec(self, obs, mode):
        with torch.no_grad():

            value, actor_actprobs = self.actor_critic(obs)
            if mode == 'run':
                c = Categorical(actor_actprobs)
                a = c.sample()
                logprob_a = c.log_prob(a)
            else:
                a = torch.argmax(actor_actprobs, dim=-1)
                logprob_a = torch.tensor(-1)  # dummy
        return value, a.unsqueeze(-1), logprob_a.unsqueeze(-1)  # 三个返回值的维度都是(threads, 1)

    def merge(self, o, a, logprob_a, adv, value_pred, r_target):
        assert o.shape[1] == 1  # 硬编码thread=1
        o = o.squeeze(1)
        a = a.reshape(-1, a.shape[-1])
        logprob_a = logprob_a.reshape(-1, logprob_a.shape[-1])
        adv = adv.reshape(-1, 1)
        value_pred = value_pred.reshape(-1, 1)
        r_target = r_target.reshape(-1, 1)
        return copy.deepcopy(o), copy.deepcopy(a), copy.deepcopy(logprob_a), copy.deepcopy(adv), copy.deepcopy(value_pred), copy.deepcopy(r_target)
        # return o, a, logprob_a, adv, value_pred, r_target

    def train(self, total_num_steps):
        self.total_num_steps = total_num_steps
        self.entropy_coef *= self.entropy_coef_decay  # start with 0.001, decay by *= 0.99

        ## get adv
        # Q 为啥i神还要再过一次网络呢? A 计算critic_loss时就不用再过网络了
        adv = self.returns[:-1] - self.value_preds[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)  # 优势标准化，一个常用trick

        '''for vecenv！from (T_horizon, threads, dim) to (T_horizon*threads, dim)'''
        o, a, logprob_a, adv, value_pred, r_target = self.merge(self.obs[:-1],
                                                                self.actions,
                                                                self.logprob_a,
                                                                adv,
                                                                self.value_preds[:-1],
                                                                self.returns[:-1])

        # Slice long trajectopy into short trajectory and perform mini-batch PPO update
        als, cls, dist_entropies = [], [], []
        optim_iter_num = int(math.ceil(o.shape[0] / self.optim_batch_size))
        for k in range(self.K_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(o.shape[0])
            np.random.shuffle(perm)
            o, a, logprob_a, adv, r_target = o[perm], a[perm], logprob_a[perm], adv[perm], r_target[perm]

            # 关键修改，actor_critic共享前层
            al, dist_entropy, cl = self.PPO_update_actor_critic(o, a, logprob_a, adv,
                                                                value_pred, r_target,
                                                                self.actor_critic, self.optimizer,
                                                                optim_iter_num)
            als.append(al)
            dist_entropies.append(dist_entropy)
            cls.append(cl)

        self.train_steps += optim_iter_num * self.K_epochs
        self.writer.add_scalar('watch/PPO/actor_critic_train_steps', self.train_steps, total_num_steps)
        self.writer.add_scalar('watch/PPO/actor_loss', np.mean(als), total_num_steps)
        self.writer.add_scalar('watch/PPO/critic_loss', np.mean(cls), total_num_steps)
        self.writer.add_scalar('watch/PPO/dist_entropy', np.mean(dist_entropies), total_num_steps)

    def PPO_update_actor_critic(self, s, a, logprob_a, adv, value_past, td_target, actor_critic, optimizer, optim_iter_num):
        als, dist_entropies, cls = [], [], []
        for j in range(optim_iter_num):
            index = slice(j * self.optim_batch_size, min((j + 1) * self.optim_batch_size, s.shape[0]))

            ## 过网络
            value_now, action_prob = actor_critic(s[index])
            ## 计算actor_loss
            distribution = Categorical(action_prob)
            logprob_a_now = distribution.log_prob(a[index].squeeze(-1)).unsqueeze(-1)  # shape = (batch, )
            dist_entropy = distribution.entropy()
            ratio = torch.exp(logprob_a_now - logprob_a[index])

            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
            surr_loss = -torch.min(surr1, surr2)
            actor_loss = torch.mean(surr_loss)
            als.append(actor_loss.item())
            dist_entropies.append(torch.mean(dist_entropy).item())

            ## 计算critic_loss
            value_now_clipped = value_past[index] + (value_now - value_past[index]).clamp(-self.clip_rate, self.clip_rate)
            value_loss_clipped = (value_now_clipped - td_target[index]).pow(2)
            value_loss = (value_now - td_target[index]).pow(2)
            critic_loss = 0.5 * torch.max(value_loss, value_loss_clipped).mean()  # 这个0.5相当于critic loss coef再减半
            cls.append(critic_loss.item())

            ## 反向传播
            optimizer.zero_grad()
            total_loss = critic_loss * self.critic_loss_coef \
                         + actor_loss \
                         - torch.mean(dist_entropy) * self.entropy_coef

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            optimizer.step()
        return np.mean(als), np.mean(dist_entropies), np.mean(cls)






