import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import numpy as np
from collections import deque

from utils import epsilon_scheduler, beta_scheduler, update_target, print_log, load_model, save_model, print_progress, save_checkpoint
from model import DQN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

def train(env, args, writer): 
    current_model = DQN(env, args).to(args.device)
    target_model = DQN(env, args).to(args.device)

    if args.noisy:
        current_model.update_noisy_modules()
        target_model.update_noisy_modules()

    if args.load_model and os.path.isfile(args.load_model):
        load_model(current_model, args)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
    beta_by_frame = beta_scheduler(args.beta_start, args.beta_frames)

    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.alpha)
    else:
        replay_buffer = ReplayBuffer(args.buffer_size)
    
    state_deque = deque(maxlen=args.multi_step)
    reward_deque = deque(maxlen=args.multi_step)
    action_deque = deque(maxlen=args.multi_step)

    optimizer = optim.Adam(current_model.parameters(), lr=args.lr)

    reward_list, length_list, loss_list = [], [], []
    episode_reward = 0
    episode_length = 0
    n_episodes = 0

    prev_time = time.time()
    prev_frame = 1

    state = env.reset()
    for frame_idx in range(1, args.max_frames + 1):
        
        print_progress(frame_idx, args.max_frames, n_episodes)

        if args.render:
            env.render()

        if args.noisy:
            current_model.sample_noise()
            target_model.sample_noise()

        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)

        next_state, reward, done, _ = env.step(action)
        state_deque.append(state)
        reward_deque.append(reward)
        action_deque.append(action)

        if len(state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(reward_deque, args.gamma)
            n_state = state_deque[0]
            n_action = action_deque[0]
            replay_buffer.push(n_state, n_action, n_reward, next_state, np.float32(done))

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            n_episodes += 1
            state = env.reset()
            reward_list.append(episode_reward)
            length_list.append(episode_length)
            if args.logging:
                writer.add_scalar("data/episode_reward", episode_reward, frame_idx)
                writer.add_scalar("data/episode_length", episode_length, frame_idx)
            episode_reward, episode_length = 0, 0
            state_deque.clear()
            reward_deque.clear()
            action_deque.clear()

        if len(replay_buffer) > args.learning_start and frame_idx % args.train_freq == 0:
            beta = beta_by_frame(frame_idx)
            loss = compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta)
            loss_list.append(loss.item())
            if args.logging:
                writer.add_scalar("data/loss", loss.item(), frame_idx)

        if frame_idx % args.update_target == 0:
            update_target(current_model, target_model)

        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, args.max_frames, n_episodes, prev_frame, prev_time, reward_list, length_list, loss_list)
            reward_list.clear(), length_list.clear(), loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()
            save_model(current_model, args)

        if args.checkpoint_freq is not None and frame_idx % args.checkpoint_freq == 0:
            save_checkpoint(current_model, args, frame_idx)


    save_model(current_model, args)


def compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta=None):
    """
    Calculate loss and optimize for non-c51 algorithm
    """
    if args.prioritized_replay:
        state, action, reward, next_state, done, weights, indices = replay_buffer.sample(args.batch_size, beta)
    else:
        state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
        weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    if not args.c51:
        q_values = current_model(state)
        target_next_q_values = target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        if args.double:
            next_q_values = current_model(next_state)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)
        else:
            next_q_value = target_next_q_values.max(1)[0]

        expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-5
        loss = (loss * weights).mean()
    
    else:
        q_dist = current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.num_atoms)
        q_dist = q_dist.gather(1, action).squeeze(1)
        q_dist.data.clamp_(0.01, 0.99)

        target_dist = projection_distribution(current_model, target_model, next_state, reward, done, 
                                              target_model.support, target_model.offset, args)

        loss = - (target_dist * q_dist.log()).sum(1)
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-6
        loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    if args.prioritized_replay:
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss


def projection_distribution(current_model, target_model, next_state, reward, done, support, offset, args):
    delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms - 1)

    target_next_q_dist = target_model(next_state)

    if args.double:
        next_q_dist = current_model(next_state)
        next_action = (next_q_dist * support).sum(2).max(1)[1]
    else:
        next_action = (target_next_q_dist * support).sum(2).max(1)[1]

    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(target_next_q_dist.size(0), 1, target_next_q_dist.size(2))
    target_next_q_dist = target_next_q_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand_as(target_next_q_dist)
    done = done.unsqueeze(1).expand_as(target_next_q_dist)
    support = support.unsqueeze(0).expand_as(target_next_q_dist)

    Tz = reward + args.gamma * support * (1 - done)
    Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
    b = (Tz - args.Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    target_dist = target_next_q_dist.clone().zero_()
    target_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_next_q_dist * (u.float() - b)).view(-1))
    target_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_next_q_dist * (b - l.float())).view(-1))

    return target_dist

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret