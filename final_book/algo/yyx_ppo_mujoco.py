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
from algo.net import Mujoco_Actor, Mujoco_Critic, Atari_Actor_Critic


class PPO():
    def __init__(self,
                 output_dir,
                 device,
                 writer,
                 use_naive_env,
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
                 a_optim_batch_size=64,
                 c_optim_batch_size=64,  # mute
                 entropy_coef=0,  # 0.001
                 entropy_coef_decay=0.9998,
                 ):

        self.output_dir = output_dir
        self.device = device
        self.writer = writer
        self.use_naive_env = use_naive_env
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
        self.a_optim_batch_size = a_optim_batch_size
        self.c_optim_batch_size = c_optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.total_num_steps = None

        self.data = []  # 经验回放池

        # 硬编码网络，for Swimmer
        if len(self.obs_space.shape) == 1 and \
                isinstance(self.act_space, gym.spaces.Box):
            self.actor = Mujoco_Actor(self.obs_space.shape[0], self.act_space.shape[0]).to(self.device)
            self.critic = Mujoco_Critic(self.obs_space.shape[0]).to(self.device)

        # 硬编码网络，for atari
        elif len(self.obs_space.shape) == 3 and \
                isinstance(self.act_space, gym.spaces.Discrete):
            self.actor = Atari_Actor_Critic(self.obs_space.shape[0], self.act_space.n).to(self.device)

        else:
            raise NotImplementedError

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        self.a_train_steps = 0
        self.c_train_steps = 0

        assert len(np.unique(self.act_space.low)) == 1
        assert len(np.unique(self.act_space.high)) == 1

    def save(self, timestep, is_evaluate, is_newbest=False):
        save_dir = os.path.join(self.output_dir, 'model')
        save_dir += '/eval' if is_evaluate else '/train'
        if is_newbest: save_dir += '/best_model'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        postfix = 'best' if is_newbest else f'ts{timestep}'
        # torch.save(self.critic[i].state_dict(), save_dir + f"/ppo_critic_{postfix}.pth")
        torch.save(self.actor.state_dict(), save_dir + f"/ppo_actor_{postfix}.pth")

    def load(self, load_dir):
        self.actor.load_state_dict(torch.load(load_dir))
        # self.critic.load_state_dict(torch.load(load_dir + f"/ppo_critic_ts{timestep}.pth"))

    def select_action_for_vec(self, obs, mode):
        # 离散动作空间
        if isinstance(self.act_space, gym.spaces.Discrete):
            actor_actprobs = self.actor(obs)
            if mode == 'run':
                c = Categorical(actor_actprobs)
                a = c.sample()
                logprob_a = c.log_prob(a)
            else:
                # TODO according to ishen, here we can use dist.mode()?
                a = torch.argmax(actor_actprobs, dim=-1)
                logprob_a = torch.tensor(-1)  # dummy
            a, logprob_a = a.unsqueeze(-1), logprob_a.unsqueeze(-1)
        # 连续动作空间
        elif isinstance(self.act_space, gym.spaces.Box):
            if mode == 'run':
                dist = self.actor.get_dist(obs)
                a = dist.sample()
                logprob_a = dist.log_prob(a)
            else:
                a = self.actor.evaluate_mode(obs)
                logprob_a = torch.tensor(-1)  # dummy
        else:
            raise NotImplementedError

        # 两个返回值的维度都是(threads, act_dim)
        return a.detach(), logprob_a.detach()

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):

        r_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, 1))
        done_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, 1))
        # TODO 下面这四个对于Swimmer OK，但对于雅达利，维度应是(T_horizon, threads, stack_frame, 84, 84)
        s_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, self.obs_space.shape[0]))
        a_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, self.act_space.shape[0]))  # 离散动作
        s_prime_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, self.obs_space.shape[0]))
        logprob_a_batch = torch.zeros((self.T_horizon, self.n_rollout_threads, self.act_space.shape[0]))  # 离散动作

        # data中除done是np.array外，其他都是torch.tensor，这是与icde的不同
        for i, transition in enumerate(self.data):  # T_horizon # self.data——
            s, a, r, s_prime, logprob_a, done = transition
            s_batch[i] = s
            s_prime_batch[i] = s_prime
            if self.use_naive_env:
                a_batch[i] = torch.FloatTensor(a).to(self.device)
                r_batch[i] = torch.FloatTensor([[r]]).to(self.device)
                logprob_a_batch[i] = torch.FloatTensor(logprob_a).to(self.device)
                done_batch[i] = torch.FloatTensor([[done]]).to(self.device)
            else:
                a_batch[i] = a
                r_batch[i] = r
                logprob_a_batch[i] = logprob_a
                done_batch[i] = torch.FloatTensor(done).unsqueeze(-1).to(self.device)

        self.data = []  # Clean history trajectory

        s_batch, a_batch, r_batch, s_prime_batch, logprob_a_batch, done_batch = \
            s_batch.to(self.device), \
            a_batch.to(self.device), \
            r_batch.to(self.device), \
            s_prime_batch.to(self.device), \
            logprob_a_batch.to(self.device), \
            done_batch.to(self.device), \

        return s_batch, a_batch, r_batch, s_prime_batch, logprob_a_batch, done_batch

    def merge(self, o, a, logprob_a, adv, r_target):
        o = o.reshape(-1, o.shape[-1])
        a = a.reshape(-1, a.shape[-1])
        logprob_a = logprob_a.reshape(-1, logprob_a.shape[-1])
        adv = adv.reshape(-1, 1)
        r_target = r_target.reshape(-1, 1)
        return o, a, logprob_a, adv, r_target

    def train(self, total_num_steps):
        self.total_num_steps = total_num_steps
        self.entropy_coef *= self.entropy_coef_decay  # start with 0.001, decay by *= 0.99

        o, a, r, o_prime, logprob_a, done = self.make_batch()

        # shape = (agent, T_horizon, threads, dim)

        '''Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        def func(ref_r, ref_critic):
            # ref_r.shape = (T_horizon, thread, 1)
            with torch.no_grad():

                '''1. 过critic，得到vs和vs_'''
                vs, vs_ = ref_critic(o), ref_critic(o_prime)
                '''2. 得到deltas'''
                deltas = ref_r + self.gamma * vs_ - vs
                deltas = deltas.squeeze(-1).cpu().numpy()  # shape = (T_horizon, threads)
                '''3. done for GAE deltas adv_list, target_list'''
                adv = [[0 for _ in range(self.n_rollout_threads)]]
                for dlt, mask in zip(deltas[::-1], done.squeeze(-1).cpu().numpy()[::-1]):  # threadsGAEadv
                    advantage = dlt + self.gamma * self.lambd * np.array(adv[-1]) * (1 - mask)
                    adv.append(list(advantage))
                adv.reverse()
                adv = copy.deepcopy(adv[0:-1])
                adv = torch.tensor(adv).unsqueeze(-1).float().to(self.device)
                target = adv + vs

            return adv, target

        '''advtd-target'''
        adv, r_target = func(r, self.critic)  # adv_list.shape = (agent, T_horizon, 1) (agent, T_horizon, threads, 1)

        '''for vecenv！from (T_horizon, threads, dim) to (T_horizon*threads, dim)'''
        o, a, logprob_a, adv, r_target = self.merge(o, a, logprob_a, adv, r_target)

        '''Kactorcritic'''
        # Slice long trajectopy into short trajectory and perform mini-batch PPO update
        als, cls, dist_entropies = [], [], []
        a_optim_iter_num = int(math.ceil(o.shape[0] / self.a_optim_batch_size))
        c_optim_iter_num = int(math.ceil(o.shape[0] / self.c_optim_batch_size))
        for k in range(self.K_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(o.shape[0])
            np.random.shuffle(perm)
            o, a, logprob_a, adv, r_target = o[perm], a[perm], logprob_a[perm], adv[perm], r_target[perm]
            al, dist_entropy = self.PPO_update_actor(o, a, logprob_a, adv, self.actor, self.actor_optimizer,
                                       a_optim_iter_num)
            cl = self.PPO_update_critic(o, r_target, self.critic, self.critic_optimizer,
                                         c_optim_iter_num)
            als.append(al)
            cls.append(cl)
            dist_entropies.append(dist_entropy)

        self.a_train_steps += a_optim_iter_num * self.K_epochs
        self.c_train_steps += c_optim_iter_num * self.K_epochs
        self.writer.add_scalar('watch/PPO/actor_train_steps', self.a_train_steps, total_num_steps)
        self.writer.add_scalar('watch/PPO/critic_train_steps', self.c_train_steps, total_num_steps)

        self.writer.add_scalar('watch/PPO/actor_loss', np.mean(als), total_num_steps)
        self.writer.add_scalar('watch/PPO/critic_loss', np.mean(cls), total_num_steps)
        self.writer.add_scalar('watch/PPO/dist_entropy', np.mean(dist_entropies), total_num_steps)

    def PPO_update_actor(self, s, a, logprob_a, adv, ref_actor, ref_actor_optimizer, a_optim_iter_num):
        als, dist_entropies = [], []
        for j in range(a_optim_iter_num):  #
            index = slice(j * self.a_optim_batch_size, min((j + 1) * self.a_optim_batch_size, s.shape[0]))
            if isinstance(self.act_space, gym.spaces.Discrete):  # 离散动作
                distribution = Categorical(ref_actor(s[index]))
                logprob_a_now = distribution.log_prob(a[index].squeeze(-1)).unsqueeze(-1)  # shape = (batch, )
                dist_entropy = distribution.entropy().unsqueeze(-1)
                ratio = torch.exp(logprob_a_now - logprob_a[index])
            elif isinstance(self.act_space, gym.spaces.Box):  # 连续动作
                distribution = ref_actor.get_dist(s[index])
                logprob_a_now = distribution.log_prob(a[index])
                dist_entropy = distribution.entropy().sum(-1, keepdim=True)
                ratio = torch.exp(logprob_a_now.sum(-1, keepdim=True) - logprob_a[index].sum(-1, keepdim=True))

            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
            surr_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
            actor_loss = torch.mean(surr_loss)
            als.append(actor_loss.item())
            dist_entropies.append(torch.mean(dist_entropy).item())
            ref_actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(ref_actor.parameters(), 40)
            ref_actor_optimizer.step()
        return np.mean(als), np.mean(dist_entropies)

    def PPO_update_critic(self, s, td_target, ref_critic, ref_critic_optimizer, c_optim_iter_num):
        cls = []
        for j in range(c_optim_iter_num):
            index = slice(j * self.c_optim_batch_size, min((j + 1) * self.c_optim_batch_size, s.shape[0]))
            critic_loss = (ref_critic(s[index]) - td_target[index]).pow(2).mean()
            for name, param in ref_critic.named_parameters():
                if 'weight' in name:  # OK
                    critic_loss += param.pow(2).sum() * self.l2_reg  # 首次debug时，+=9.3 += 9.9 += 0.07 是不是太大了？
            cls.append(critic_loss.item())  # 首次debug时，critic_loss = 20
            ref_critic_optimizer.zero_grad()
            critic_loss.backward()
            ref_critic_optimizer.step()
        return np.mean(cls)






