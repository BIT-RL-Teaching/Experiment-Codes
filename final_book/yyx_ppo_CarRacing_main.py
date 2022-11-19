import copy
import glob
import os
import time
from collections import deque
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.utils import update_linear_schedule
from arguments import get_args
# from envs.envs import make_vec_envs
from algo.yyx_ppo_CarRacing import PPO_CarRacing
from utils import *
# from a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate

from tensorboardX import SummaryWriter

import sys
import os

sys_path_name = os.path.dirname(os.path.dirname(__file__))
sys.path.append(sys_path_name)
print('-------', sys_path_name)
# from yyx_common_tools.common import *

def Action_adapter(a):
    # for CarRacing
    return a * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])


def evaluate_policy(env, agent, args, device, render=False):
    scores = 0
    turns = 5
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0

        while not done:
            # if args.eval or args.debug: print('steps = ', steps)
            # Take deterministic actions at test time

            s = torch.FloatTensor(s).unsqueeze(0).to(device)
            _, a, _ = agent.select_action_for_vec(s, mode='eval')
            a = a.cpu().detach().numpy()
            # print('a = ', a)
            act = Action_adapter(a)
            s_prime, r, done, info = env.step(act.squeeze(0))
            if args.env_name == 'Pendulum-v0': s_prime = s_prime.squeeze(-1)
            ep_r += r
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
        print('scores = ', scores)
    return scores / turns


def main(args):
    args.env_name = 'CarRacing-v0'
    # for CarRacing's Env
    args.frames_stack = 4
    args.action_repeat = 8

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.debug:
        args.num_env_steps = 3000
        args.num_processes = 1
        args.num_steps = 50  # 也即T_horizon
        # num_updates = num_env_steps // num_processes // num_steps, 也即num_updates = 10
        args.save_interval = 5
        args.eval_interval = 2
        args.eval_episodes = 1
        args.log_interval = 2
        args.output_dir = 'runs/debug'

    assert args.num_processes == 1

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print(f'when args.debug = {args.debug}, num_updates = {num_updates}')
    if args.use_linear_lr_decay:
        print('使用学习率衰减')

    args.output_dir += f'/{args.env_name}'
    args.output_dir += f'/{datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")}'
    if args.tmp:
        args.output_dir += '_TMP'
    if args.exp_postfix != '':
        args.output_dir += f'_{args.exp_postfix}'
    if args.l2_reg == 0.0:
        args.output_dir += '_NotUseL2Reg'
    if args.l2_reg != 1e-3:
        args.output_dir += f'_L2Reg={args.l2_reg}'
    if args.entropy_coef != 1e-3:
        args.output_dir += f'_Entropy={args.entropy_coef}'
    if not args.vital_debug:
        args.output_dir += f'_NotVitaldebug'
    if args.debug_dont_compensate_die_state:
        args.output_dir += f'_DebugDontDieState'
    if args.debug_use_smooth_l1_loss:
        args.output_dir += f'_DebugUseSmoothL1Loss'



    if not args.eval:
        writer = SummaryWriter(args.output_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_dir = args.output_dir + "/model"

    torch.set_num_threads(1)  # 避免在服务器占用过多资源
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('----yyx: device = ', device)

    if args.use_naive_env:
        from envs.yyx_carracing_env import Env
        envs = Env(args)
        eval_envs = Env(args)
        # envs = gym.make(args.env_name)
        # eval_envs = gym.make(args.env_name)
    else:
        raise NotImplementedError


    batch_size = args.num_steps * args.num_processes // args.num_mini_batch
    print('----yyx: batch_size = ', batch_size)
    kwargs = {
        "args": args,
        "output_dir": args.output_dir,
        "device": device,
        "writer": writer if not args.eval else None,
        "T_horizon": args.num_steps,
        "n_rollout_threads": args.num_processes,
        "obs_space": envs.observation_space,
        "act_space": envs.action_space,
        "gamma": args.gamma,
        "lambd": args.gae_lambda,  # For GAE
        "clip_rate": args.clip_param,  # 0.2
        "K_epochs": args.K_epochs,
        "a_lr": args.a_lr,
        "c_lr": args.c_lr,
        "l2_reg": args.l2_reg,  # L2 regulization for Critic
        "optim_batch_size": batch_size,
        "entropy_coef": args.entropy_coef,  # Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay": args.entropy_coef_decay,
        "critic_loss_coef": args.critic_loss_coef,
        "max_grad_norm": args.max_grad_norm,
    }

    agent = PPO_CarRacing(**kwargs)

    if args.eval:
        assert args.load_dir is not None
        agent.load(args.load_dir)
        score = evaluate_policy(envs, agent, args, device, render=True)
        print(f'env: {args.env_name}, eval score: {score}')
        exit(0)

    episode_rewards = deque(maxlen=10)
    obs = envs.reset()
    obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

    total_steps = 0
    eval_best_ep_reward = float('-inf')
    ep_r = 0
    start = time.time()
    for j in range(num_updates):
        total_steps = (j + 1) * args.num_processes * args.num_steps

        if args.use_linear_lr_decay:
            assert args.a_lr == args.c_lr
            # decrease learning rate linearly
            update_linear_schedule(agent.optimizer, None, j, num_updates, args.a_lr)

        for step in range(args.num_steps):
            # print('step = ', step)
            value, a, logprob_a = agent.select_action_for_vec(obs, mode='run')  # 关键代码一 sample actions
            a = a.cpu().detach().numpy()
            logprob_a = logprob_a.cpu().detach().numpy()
            act = Action_adapter(a)
            try:
                obs_prime, r, done, infos = envs.step(act.squeeze(0))  # 关键代码二 状态转移
            except:
                print(1)
            obs_prime = torch.FloatTensor(obs_prime).unsqueeze(0).to(device)
            data = (obs, obs_prime, a, logprob_a, r, done, value)

            agent.put_data(data)
            obs = obs_prime
            ep_r += r
            if done:
                obs = envs.reset()  # 很关键
                obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
                episode_rewards.append(ep_r)
                ep_r = 0



        # train
        if args.debug: print('agent.train()')
        agent.train(total_steps)

        # routinely save model
        if (j % args.save_interval == 0
            or j == num_updates - 1) and save_dir != "":
            save_path = os.path.join(save_dir, 'PPO')
            agent.save(total_steps, is_evaluate=False)
            print(f'successfully save model to {save_path}')

        # log
        if j % args.log_interval == 0 and len(episode_rewards) >= 1:
            if args.debug: print('---------log---------')
            end = time.time()

            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_steps,
                            int(total_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards)))

            writer.add_scalar('train/mean_episode_reward', np.mean(episode_rewards), total_steps)
            writer.add_scalar('train/min_episode_reward', np.min(episode_rewards), total_steps)
            writer.add_scalar('train/max_episode_reward', np.max(episode_rewards), total_steps)

        # evaluate
        if args.eval_interval is not None and j % args.eval_interval == 0:
            if args.vital_debug:
                score = evaluate_policy(eval_envs, agent, args, device)
            else:
                score = evaluate_policy(envs, agent, args, device)
            writer.add_scalar('eval/episode_reward', score, total_steps)
            if score > eval_best_ep_reward:
                eval_best_ep_reward = score
                agent.save(total_steps, is_evaluate=True, is_newbest=True)
                print(f'In step = {total_steps}, successfully save eval best model, score = {score}')


    print('OK！')
    envs.close()



if __name__ == "__main__":
    args = get_args()

    try:
        main(args)
    except Exception as ex:
        import traceback

        print(traceback.format_exc())
        if not args.debug and not args.eval:
            send_an_error_message(program_name=__file__.split('\\')[-1],
                                  error_name=repr(ex),
                                  error_detail=traceback.format_exc())
