import time
import copy
import os
import shutil
# from algo.utils import normalize
import torch.nn.functional as F

import gym.spaces
import numpy as np
import torch
from torch.distributions import Categorical
import math
from algo.net import Mujoco_Actor, Mujoco_Critic, Atari_Actor_Critic, CarRacing_Actor_Critic


class PPO_CarRacing():
    def __init__(self,
                 args,
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
                 critic_loss_coef=0.5,
                 max_grad_norm=0.5,
                 ):
        self.args = args
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
        self.optim_batch_size = optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.critic_loss_coef = critic_loss_coef
        self.max_grad_norm = max_grad_norm
        self.total_num_steps = None

        self.data = []  # 经验回放池

        # 硬编码网络，for CarRacing
        if len(self.obs_space.shape) == 3 and \
                isinstance(self.act_space, gym.spaces.Box):
            self.actor_critic = CarRacing_Actor_Critic(self.obs_space.shape[0], self.act_space.shape[0]).to(self.device)
        else:
            raise NotImplementedError

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.a_lr)

        self.a_train_steps = 0
        self.c_train_steps = 0

    def save(self, timestep, is_evaluate, is_newbest=False):
        save_dir = os.path.join(self.output_dir, 'model')
        save_dir += '/eval' if is_evaluate else '/train'
        if is_newbest: save_dir += '/best_model'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        postfix = 'best' if is_newbest else f'ts{timestep}'
        torch.save(self.actor_critic.state_dict(), save_dir + f"/ppo_actor_critic_{postfix}.pth")

    def load(self, load_dir):
        self.actor_critic.load_state_dict(torch.load(load_dir))

    def select_action_for_vec(self, obs, mode):
        with torch.no_grad():
            if mode == 'run':
                value, dist = self.actor_critic.get_dist(obs)
                a = dist.sample()
                logprob_a = dist.log_prob(a)
            else:
                value, a = self.actor_critic.mode(obs)
                logprob_a = torch.tensor(-1)  # dummy
        # 两个返回值的维度都是(threads, act_dim)
        return value, a, logprob_a

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):

        s_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, *self.obs_space.shape))
        s_prime_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, *self.obs_space.shape))
        a_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, self.act_space.shape[0]))
        logprob_a_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, self.act_space.shape[0]))
        r_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, 1))
        done_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, 1))
        value_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, 1))

        # data中除done是np.array外，其他都是torch.tensor，这是与icde的不同
        for i, transition in enumerate(self.data):  # T_horizon # self.data——
            s, s_prime, a, logprob_a, r, done, value = transition
            s_batch[i] = s
            s_prime_batch[i] = s_prime
            a_batch[i] = torch.FloatTensor(a).to(self.device)
            logprob_a_batch[i] = torch.FloatTensor(logprob_a).to(self.device)
            r_batch[i] = torch.FloatTensor([[r]]).to(self.device)
            done_batch[i] = torch.FloatTensor([[done]]).to(self.device)
            value_batch[i] = value

        self.data = []  # Clean history trajectory

        s_batch, a_batch, r_batch, s_prime_batch, logprob_a_batch, done_batch, value_batch = \
            s_batch.to(self.device), \
            a_batch.to(self.device), \
            r_batch.to(self.device), \
            s_prime_batch.to(self.device), \
            logprob_a_batch.to(self.device), \
            done_batch.to(self.device), \
            value_batch.to(self.device)

        return s_batch, a_batch, r_batch, s_prime_batch, logprob_a_batch, done_batch, value_batch

    def merge(self, o, a, logprob_a, adv, value, r_target):
        o = o.reshape(-1, *self.obs_space.shape)
        a = a.reshape(-1, a.shape[-1])
        logprob_a = logprob_a.reshape(-1, logprob_a.shape[-1])
        adv = adv.reshape(-1, 1)
        value = value.reshape(-1, 1)
        r_target = r_target.reshape(-1, 1)
        return o, a, logprob_a, adv, value, r_target

    def train(self, total_num_steps):
        self.total_num_steps = total_num_steps
        self.entropy_coef *= self.entropy_coef_decay  # start with 0.001, decay by *= 0.99

        o, a, r, o_prime, logprob_a, done, value_pred = self.make_batch()
        # shape = (agent, T_horizon, threads, dim)

        '''Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        def func(ref_r, ref_critic):
            # ref_r.shape = (T_horizon, thread, 1)
            with torch.no_grad():
                # 1. 过critic，得到vs_
                next_value = ref_critic(o_prime.squeeze(1))[0].unsqueeze(1)
                # 2. 得到deltas
                deltas = ref_r + self.gamma * next_value * (1 - done) - value_pred
                deltas = deltas.squeeze(-1).cpu().numpy()  # shape = (T_horizon, threads)
                '''3. done for GAE deltas adv_list, target_list'''
                adv = [[0 for _ in range(self.n_rollout_threads)]]
                for dlt, mask in zip(deltas[::-1], done.squeeze(-1).cpu().numpy()[::-1]):  # threadsGAEadv
                    advantage = dlt + self.gamma * self.lambd * np.array(adv[-1]) * (1 - mask)
                    adv.append(list(advantage))
                adv.reverse()
                adv = copy.deepcopy(adv[0:-1])
                adv = torch.tensor(adv).unsqueeze(-1).float().to(self.device)
                target = adv + value_pred
            return adv, target

        '''advtd-target'''
        adv, r_target = func(r, self.actor_critic)  # adv_list.shape = (agent, T_horizon, 1) (agent, T_horizon, threads, 1)

        '''for vecenv！from (T_horizon, threads, dim) to (T_horizon*threads, dim)'''
        o, a, logprob_a, adv, value_pred, r_target = self.merge(o, a, logprob_a, adv, value_pred, r_target)

        # Slice long trajectopy into short trajectory and perform mini-batch PPO update
        als, cls, dist_entropies = [], [], []
        optim_iter_num = int(math.ceil(o.shape[0] / self.optim_batch_size))
        for k in range(self.K_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(o.shape[0])
            np.random.shuffle(perm)
            o, a, logprob_a, adv, r_target = o[perm], a[perm], logprob_a[perm], adv[perm], r_target[perm]
            al, dist_entropy, cl = self.PPO_update_actor_critic(o, a, logprob_a, adv,
                                                            value_pred, r_target,
                                                            self.actor_critic, self.optimizer,
                                                            optim_iter_num)
            # cl = self.PPO_update_critic(o, r_target, self.critic, self.critic_optimizer,
            #                              c_optim_iter_num)
            als.append(al)
            cls.append(cl)
            dist_entropies.append(dist_entropy)

        if self.args.debug: print('critic_loss:', np.mean(cls))
        self.writer.add_scalar('watch/PPO/actor_loss', np.mean(als), total_num_steps)
        self.writer.add_scalar('watch/PPO/critic_loss', np.mean(cls), total_num_steps)
        self.writer.add_scalar('watch/PPO/dist_entropy', np.mean(dist_entropies), total_num_steps)

    def PPO_update_actor_critic(self, s, a, logprob_a, adv, value_past, td_target, actor_critic, optimizer, optim_iter_num):
        als, dist_entropies, cls = [], [], []
        for j in range(optim_iter_num):
            index = slice(j * self.optim_batch_size, min((j + 1) * self.optim_batch_size, s.shape[0]))
            if isinstance(self.act_space, gym.spaces.Box):  # 连续动作
                value_now, distribution = actor_critic.get_dist(s[index])
                logprob_a_now = distribution.log_prob(a[index])
                dist_entropy = distribution.entropy().sum(-1, keepdim=True)
                ratio = torch.exp(logprob_a_now.sum(-1, keepdim=True) - logprob_a[index].sum(-1, keepdim=True))
            else:
                raise NotImplementedError

            ## 计算actor_loss
            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
            surr_loss = -torch.min(surr1, surr2)
            actor_loss = torch.mean(surr_loss)
            als.append(actor_loss.item())
            dist_entropies.append(torch.mean(dist_entropy).item())

            if self.args.debug_use_smooth_l1_loss:
                value_loss = F.smooth_l1_loss(value_now, td_target[index])
                critic_loss = 0.5 * value_loss.mean()
            else:
                ## 计算critic_loss，默认使用clipped value loss
                ## 意义：not clear yet
                # TODO when critic_loss > 500, value_now = 0.5, td_target = 50, so value_loss = 2500
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







