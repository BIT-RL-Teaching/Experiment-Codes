import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,  # yyx修改，源码中设置为None，也即不eval
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 1 与i神ppo保持一致)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=2e6,
        help='number of environment steps to train')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=2048,
        help='number of forward steps in A2C (default: 5)')  # 也即T_horizon
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_false',
        default='true',
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--critic-loss-coef',
        type=float,
        default=0.5,
        help='critic loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')

    parser.add_argument('--K_epochs', type=int, default=10)
    parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Entropy coefficient of Actor')
    parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='')

    # yyx_add
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--exp_postfix', type=str, default='')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='runs')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render_sleep_secs', type=float, default=0.01)
    parser.add_argument('--use_naive_env', action='store_false')
    parser.add_argument('--cuda', action='store_false', default=True, help='disables CUDA training')
    parser.add_argument('--gpu_id', type=str, default='0', help='')
    parser.add_argument('--vital_debug', action='store_false')
    parser.add_argument('--debug_dont_compensate_die_state', action='store_true')
    parser.add_argument('--debug_use_smooth_l1_loss', action='store_false')
    parser.add_argument('--debug_use_ppo2_value_loss_clip', action='store_false')

    # postprocess
    parser.add_argument('--gen_img_raw_for_postprocess', action='store_true')
    parser.add_argument('--do_pong_postprocess', action='store_true')
    parser.add_argument('--eval_early_stop', type=int, default=0)


    args = parser.parse_args()

    return args
