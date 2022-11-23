import json
import time
from collections import deque

import numpy as np
import torch

from algo.utils import update_linear_schedule
from arguments import get_args
from envs.envs import make_vec_envs
from algo.yyx_ppo_atari import PPO_Atari
from utils import *
# from a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from PIL import Image
import sys
import os

from postprocess.locate_ball_board import locate_ball, locate_board

sys_path_name = os.path.dirname(os.path.dirname(__file__))
sys.path.append(sys_path_name)
print('-------', sys_path_name)
# from yyx_common_tools.common import *

distances = []
act_entropies = []

def draw_dis_entropy_correlation():
    # plt.plot(distances, act_entropies)
    plt.scatter(distances, act_entropies)
    plt.xlabel('distance between ball and board')
    plt.ylabel('action entropy')
    plt.show()


def evaluate_policy(env, agent, args, render=False):
    scores = 0
    turns = 1
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not done:
            if args.eval_early_stop and steps > args.eval_early_stop: break
            if (args.eval or args.debug) and steps % 100 == 0: print('steps = ', steps)
            if args.gen_img_raw_for_postprocess:
                img = Image.fromarray(s[0][0].cpu().numpy())
                # convert('L') 意为转为每个像素由八bit存储的灰度图
                img.convert('L').save(args.output_dir + f'/img_raw/{steps}.png')
            if args.do_pong_postprocess:  #
                pos_ball = locate_ball(s[0][0].cpu().numpy())
                pos_board = locate_board(s[0][0].cpu().numpy())
                dis = np.linalg.norm(np.array(pos_ball) - np.array(pos_board))
                entropy = agent.get_act_entropy(s)
                distances.append(dis)
                act_entropies.append(entropy)

            _, a, _ = agent.select_action_for_vec(s, mode='eval')
            # print('a = ', a)
            s, r, done, info = env.step(
                a.cpu() * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]) if args.env_name == 'CarRacing-v0' else a
            )
            ep_r += r
            steps += 1
            if render:
                env.render()
                if args.render_sleep_secs != 0: time.sleep(args.render_sleep_secs)
        scores += ep_r
    return scores / turns


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.debug:
        args.num_env_steps = 3000
        args.num_processes = 1
        args.num_steps = 64  # 也即T_horizon
        args.num_mini_batch = 4  # 也即batch_size = 16
        args.save_interval = 5
        args.eval_interval = 10
        args.eval_episodes = 1
        args.log_interval = 2
        args.output_dir = 'runs/debug'

    if args.gen_img_raw_for_postprocess:
        args.output_dir = 'runs/postprocess'

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
    if not args.debug_use_ppo2_value_loss_clip:
        args.output_dir += f'_DebugNotUseppo2ValueLossClip'

    save_dir = args.output_dir + "/model"

    if not args.eval:
        writer = SummaryWriter(args.output_dir)
    if args.gen_img_raw_for_postprocess and not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(1)  # 避免在服务器占用过多资源
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('----yyx: device = ', device)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print(f'when args.debug = {args.debug}, num_updates = {num_updates}')
    if args.use_linear_lr_decay:
        print('使用学习率衰减')

    batch_size = args.num_steps * args.num_processes // args.num_mini_batch
    assert args.a_lr == args.c_lr
    lr = args.a_lr
    print('----yyx: batch_size = ', batch_size)

    # yyx: add for car-racing (status: deprecated, now we use yyx_ppo_CarRacing_main.py!)
    if args.env_name == 'CarRacing-v0':
        args.frames_stack = 4
        args.action_repeat = 8

    # hook the params
    if os.path.exists(args.output_dir):
        with open(os.path.join(args.output_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(vars(args)))

    dummy_env = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
                         None, device, False, args=args)

    kwargs = {
        "args": args,
        "output_dir": args.output_dir,
        "device": device,
        "writer": writer if not args.eval else None,
        "T_horizon": args.num_steps,
        "n_rollout_threads": args.num_processes,
        "obs_space": dummy_env.observation_space,  # 硬编码
        "act_space": dummy_env.action_space,
        "gamma": args.gamma,
        "lambd": args.gae_lambda,  # For GAE
        "clip_rate": args.clip_param,  # 0.2
        "K_epochs": args.K_epochs,
        "a_lr": lr,
        "c_lr": lr,
        "l2_reg": args.l2_reg,  # L2 regulization for Critic
        "optim_batch_size": batch_size,
        "entropy_coef": args.entropy_coef,  # Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay": args.entropy_coef_decay,
        "eps": args.eps,
        "critic_loss_coef": args.critic_loss_coef,
        "max_grad_norm": args.max_grad_norm,
    }

    agent = PPO_Atari(**kwargs)

    if args.eval:
        eval_envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
                                  None, device, False, is_eval=True, args=args)
        assert args.load_dir is not None
        agent.load(args.load_dir)
        score = evaluate_policy(eval_envs, agent, args, render=True)
        print(f'env: {args.env_name}, eval score: {score}')

        draw_dis_entropy_correlation()
        exit(0)

    if args.load_dir is not None:  # 加载预训练模型续杯训练
        print('Load the pretrained model, continuous to train!')
        agent.load(args.load_dir)

    # 对于allow_early_resets, 为啥train时是False，eval时是True？先和i神保持一致
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
                         args.output_dir, device,
                         allow_early_resets=False, args=args)
    eval_envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
                              args.output_dir, device,
                              allow_early_resets=True, is_eval=True, args=args)

    obs = envs.reset()
    episode_rewards = deque(maxlen=10)

    total_steps = 0
    eval_best_ep_reward = float('-inf')
    start = time.time()
    for j in range(num_updates):
        print(f'Updates {j}')
        total_steps = (j + 1) * args.num_processes * args.num_steps

        if args.use_linear_lr_decay:
            assert args.a_lr == args.c_lr
            # decrease learning rate linearly
            update_linear_schedule(
                agent.optimizer, None, j, num_updates, args.a_lr)

        for step in range(args.num_steps):
            # print('step = ', step)
            value, a, logprob_a = agent.select_action_for_vec(obs, mode='run')  # 关键代码一 sample actions
            obs, r, done, infos = envs.step(  # 关键代码二 状态转移
                a.cpu() * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]) if args.env_name == 'CarRacing-v0' else a
            )
            # i神式统计episode_reward，因为每个向量环境done的时机不相同
            for info in infos:
                if 'episode' in info.keys():
                    if args.debug: print('---- episode done! ----')
                    episode_rewards.append(info['episode']['r'])

            # atari的一个关键改变，data中不存obs_prime
            agent.rollout_insert(obs, a, logprob_a, value, r, done)

        agent.compute_return()
        agent.train(total_steps)
        agent.after_update()

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
            # eval需要新开一个eval_envs啊！！！
            score = evaluate_policy(eval_envs, agent, args)
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
