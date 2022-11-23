import os
import pickle
from datetime import datetime
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import wrappers
from atari_wrappers import wrap_deepmind

from agent import DQNAgent
import torch
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(4)


def train(writer,
          n_episodes=100,
          max_t=10000,
          eps_start=1.0,
          eps_end=0.05,
          eps_decay_step=1000000):
    scores = []  # 用于保存所有游戏的得分
    max_score = 0.  # 计算最大得分
    done_timesteps = []  # 记录每局游戏结束时使用的时间步
    scores_window = deque(maxlen=20)  # 记录近20局游戏的得分，用于计算平均值
    q = deque(maxlen=4)  # 用于叠放连续4帧画面的队列
    eps_decay = (eps_start - eps_end) / eps_decay_step  # 每步的epsilon衰减量
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()  # 重置游戏环境
        score = 0.
        for _ in range(4):  # 初始时填充队列
            q.append(state)
            next_state, _, _, _ = env.step(0)
            state = next_state

        for timestep in range(max_t):
            state = torch.tensor(np.array(q))  # 将队列中的4帧画面转换为tensor
            action = agent.act(state, eps)  # 使用评估网络计算动作
            next_state, reward, done, _ = env.step(action)  # 在环境中执行动作
            q.append(next_state)

            # 保存当前样本，进入模型训练过程
            agent.step(state, action, reward, torch.tensor(np.array(q)), done)
            score += reward  # 更新本局游戏的总得分
            if done:  # 如果游戏结束，记录本局相关信息
                done_timesteps.append(timestep)
                scores_window.append(score)
                scores.append(score)

                if score > max_score:
                    max_score = score
                    torch.save(agent.qnetwork_local.state_dict(),
                               SAVE_DIR + '/best_model.pth')

                print('\rEpisode {}\tTimestep {}\tScore {:.2f}\tAverage Score {:.2f}\tEps {:.2f}'.format(
                    i_episode, timestep, score, np.mean(scores_window), eps))
                writer.add_scalar('scores/eps_score', score, i_episode)
                break
            eps = max(eps-eps_decay, eps_end)

            if timestep % SAVE_EVERY == 0:
                # save the final network
                torch.save(agent.qnetwork_local.state_dict(),
                           SAVE_DIR + '/model.pth')

                # save the final scores
                with open(SAVE_DIR + '/scores', 'wb') as fp:
                    pickle.dump(scores, fp)

                # save the done timesteps
                with open(SAVE_DIR + '/dones', 'wb') as fp:
                    pickle.dump(done_timesteps, fp)

    # save the final network
    torch.save(agent.qnetwork_local.state_dict(), SAVE_DIR + '/final_model.pth')


    # save the final scores
    with open(SAVE_DIR + '/scores', 'wb') as fp:
        pickle.dump(scores, fp)

    # save the done timesteps
    with open(SAVE_DIR + '/dones', 'wb') as fp:
        pickle.dump(done_timesteps, fp)

    return scores


def test(env, trained_agent, n_games=5, n_steps_per_game=10000):
    q = deque(maxlen=4)
    scores = []
    for game in range(n_games):
        env = wrappers.Monitor(env,
                               "./test/game-{}".format(game),
                               force=True)

        observation = env.reset()
        for _ in range(4):
            q.append(observation)

        score = 0
        for step in range(n_steps_per_game):
            # env.render()
            action = trained_agent.act(torch.tensor(np.array(q)))
            observation_, reward, done, info = env.step(action)
            q.append(observation_)
            score += reward
            if done:
                print('GAME-{} OVER! score={}'.format(game, score))
                scores.append(score)
                break
        env.close()
    print('avg score of {} games: {:.2f}'.format(int(n_games), np.mean(scores)))
    print('max score: {:.2f}'.format(np.max(scores)))


if __name__ == '__main__':
    TRAIN = True  # train or test
    BUFFER_SIZE = int(1e6)  # 经验回放池容量
    BATCH_SIZE = 64  # 在训练时每次从经验池中采样的样本数量
    GAMMA = 0.99  # 折扣率
    LR = 5e-4  # 学习率
    EPS_START = 1.  # epsilon的初始值为1
    EPS_END = 0.05  # 最小值为0.05，当衰减到0.05后就不再减小
    EPS_DECAY_STEP = 1000000  # 利用前1000000步训练来进行eps衰减

    UPDATE_EVERY = 1            # 每隔多少步训练，来优化一次评估网络
    TARGET_UPDATE_EVERY = 200   # 每隔多少步训练，来优化一次策略网络
    SAVE_EVERY = 5000           # 每隔多少步训练，保存一次模型
    MAX_TIMESTEPS = 10000       # 一局游戏中最大的时间步数
    N_EPISODES = 15000          # 训练总局数

    env = gym.make('SpaceInvaders-v0')

    if TRAIN:
        # 环境初始化，wrap_deepmind方法提供了gym环境的自定义封装
        # 全部生命都耗尽之后，一个episode结束
        # 将奖励值裁切，正零负奖励值分别对应+1,0,-1
        # 其中还包括了对于输入游戏画面的裁切
        env = wrap_deepmind(env, episode_life=False, clip_rewards=True)

        # tensorboard设置
        date = f'/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        SAVE_DIR = "train"
        SAVE_DIR += date
        output_dir = 'runs/SpaceInvaders-v0'
        output_dir += date

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        writer = SummaryWriter(output_dir)
        # 初始化智能体
        agent = DQNAgent(action_size=env.action_space.n,
                         seed=0,
                         lr=LR,
                         gamma=GAMMA,
                         buffer_size=BUFFER_SIZE,
                         batch_size=BATCH_SIZE,
                         update_every=UPDATE_EVERY,
                         target_update_every=TARGET_UPDATE_EVERY)
        # 开始训练
        scores = train(writer=writer, n_episodes=N_EPISODES, max_t=MAX_TIMESTEPS,
                       eps_start=EPS_START, eps_end=EPS_END, eps_decay_step=EPS_DECAY_STEP)

        # 计算平均得分
        N = 20
        plt.plot(
            np.convolve(np.array(scores), np.ones((N, )) / N, mode='valid'))
        plt.ylabel('Score')
        plt.xlabel('Timestep')
        plt.show()
    else:
        env = wrap_deepmind(env, clip_rewards=False)

        N_GAMES = 10
        N_STEPS_PER_GAME = 10000

        # init a new agent
        trained_agent = DQNAgent(action_size=env.action_space.n, seed=0)
        trained_agent.qnetwork_local.load_state_dict(
            torch.load(r'best_model.pth', map_location="cuda:0"))
        trained_agent.qnetwork_local.eval()

        # 开始测试
        test(env, trained_agent, n_games=N_GAMES, n_steps_per_game=N_STEPS_PER_GAME)
