import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import time, os
from tensorboardX import SummaryWriter

from utils import create_log_dir, print_args, set_global_seeds
from wrappers import make_atari, wrap_atari_dqn
from arguments import get_args
from train import train
from test import test

def main():
    args = get_args()
    print_args(args)

    log_dir = create_log_dir(args)
    if not args.evaluate and args.logging:
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    env = make_atari(args.env)
    # env = wrap_atari_dqn(env, args)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        test(env, args)
        env.close()
        return

    train(env, args, writer)
    env.close()

    if args.logging:
        writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
        writer.close()


if __name__ == "__main__":
    main()
