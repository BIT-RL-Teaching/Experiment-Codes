import math
import os
import datetime
import time
import pathlib
import random

import torch
import numpy as np

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def epsilon_scheduler(eps_start, eps_final, eps_decay):
    def function(frame_idx):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)
    return function

def beta_scheduler(beta_start, beta_frames):
    def function(frame_idx):
        return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    return function

def create_log_dir(args):
    log_dir = ""
    if args.multi_step != 1:
        log_dir = log_dir + "{}-step-".format(args.multi_step)
    if args.c51:
        log_dir = log_dir + "c51-"
    if args.prioritized_replay:
        log_dir = log_dir + "per-"
    if args.dueling:
        log_dir = log_dir + "dueling-"
    if args.double:
        log_dir = log_dir + "double-"
    if args.noisy:
        log_dir = log_dir + "noisy-"
    log_dir = log_dir + "dqn-"
    
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + now

    log_dir = os.path.join("runs", log_dir)
    return log_dir

def print_progress(frame, max_frames, n_episodes):
    print("\rFrame: {:>8} / {}\tEpisode: {}".format(frame, max_frames, n_episodes), end="")

def print_log(frame, max_frames, n_episodes, prev_frame, prev_time, reward_list, length_list, loss_list):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    avg_reward = np.mean(reward_list)
    avg_length = np.mean(length_list)
    avg_loss = np.mean(loss_list) if len(loss_list) != 0 else 0.

    print("\r" + " " * 50, end="")
    print("\rFrame: {:>8} / {}\tEpisode: {}\tFPS: {:.2f}\tAvg. Reward: {:.2f}\tAvg. Length: {:.2f}\tAvg. Loss: {:.6f}".format(
        frame, max_frames, n_episodes, fps, avg_reward, avg_length, avg_loss
    ))

def print_args(args):
    print("Arguments\n=============")
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("\nGame: ", args.env)
    print("Model: {}\n".format(model_name(args)))

def model_name(args):
    fname = ""
    if args.multi_step != 1:
        fname += "{}-step-".format(args.multi_step)
    if args.c51:
        fname += "c51-"
    if args.prioritized_replay:
        fname += "per-"
    if args.dueling:
        fname += "dueling-"
    if args.double:
        fname += "double-"
    if args.noisy:
        fname += "noisy-"
    fname += "dqn"
    return fname
    
def save_checkpoint(model, args, checkpoint_id):
    checkpoint_folder = "checkpoints/{}-{}".format(model_name(args), args.save_model)
    fname = os.path.join(checkpoint_folder, f"{checkpoint_id}.pth")

    pathlib.Path(checkpoint_folder).mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), fname)
    
def save_model(model, args):
    fname = "{}-{}.pth".format(model_name(args), args.save_model)
    fname = os.path.join("models", fname)

    pathlib.Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), fname)

def load_model(model, args):
    if args.load_model is not None:
        fname = args.load_model
    else:
        fname = ""
        if args.multi_step != 1:
            fname += "{}-step-".format(args.multi_step)
        if args.c51:
            fname += "c51-"
        if args.prioritized_replay:
            fname += "per-"
        if args.dueling:
            fname += "dueling-"
        if args.double:
            fname += "double-"
        if args.noisy:
            fname += "noisy-"
        fname += "dqn-{}".format(args.save_model)
    if args.load_checkpoint is None:
        fname = os.path.join("models", fname + ".pth")
    else:
        fname = os.path.join("checkpoints", fname, args.load_checkpoint + ".pth")

    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None
    
    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    model.load_state_dict(torch.load(fname, map_location))

def set_global_seeds(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)
