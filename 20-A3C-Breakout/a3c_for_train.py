from __future__ import print_function
import torch, time, gym, argparse, sys
import numpy as np
from scipy.signal import lfilter # lfilter实现FIR或IIR滤波器
from scipy.misc import imresize  # scipy1.3.0之后不再包含imresize
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
parser.add_argument('--processes', default=8, type=int, help='number of processes')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--max_frame', default=4e7, type=int, help='max frame in exploration')
parser.add_argument('--update_freq', default=4, type=int, help='every 4 frames update once')
parser.add_argument('--max_frame_episode', default=1e4, type=int, help='random seed')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# lambda函数,自变量x,gamma,discount是函数名,冒号后面是函数体
discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]
prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.

class NNPolicy(nn.Module):
    def __init__(self, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d( 1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # ???
        self.gru = nn.GRUCell(32 * 5 * 5, 256)
        self.critic_net, self.actor_net = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def forward(self, inputs, train=True, hard=False):
        # ???
        inputs, hx = inputs           # [1,1,80,80],[1,256]
        x = F.elu(self.conv1(inputs)) # [1,32,40,40]
        x = F.elu(self.conv2(x))      # [1,32,20,20]
        x = F.elu(self.conv3(x))      # [1,32,10,10]
        x = F.elu(self.conv4(x))      # [1,32,5,5]
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx)) # [1,256]
        return self.critic_net(hx), self.actor_net(hx), hx


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

def loss_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma)           # 实现了GAE
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    entropy_loss = -(-logps * torch.exp(logps)).sum() # 最大化动作熵，类似SAC，这里更看做是个正则项
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
def worker(shared_model, shared_optimizer, rank, args, info):
    print(rank,'begin')
    # 创建环境，设置种子
    env = gym.make(args.env)
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    # 初始化模型
    model = NNPolicy(num_actions=args.num_actions)
    # 初始状态
    state = torch.tensor(prepro(env.reset()))
    # 该线程开始时间
    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True

    while info['frames'][0] <= args.max_frame: # 所有线程的actor最大探索4000万帧
        model.load_state_dict(shared_model.state_dict()) # 导入主模型参数
        hx = torch.zeros(1, 256) if done else hx.detach()
        values, logps, actions, rewards = [], [], [], []

        for _ in range(args.update_freq): # 每4个frame更新1次
            episode_length += 1
            value, logit, hx = model((state.view(1, 1, 80, 80), hx))
            
            logp = F.log_softmax(logit, dim=-1) # 做完softmax再取对数

            action = torch.exp(logp).multinomial(num_samples=1).data[0] # 按概率从actions中sample
            state, reward, done, _ = env.step(action.numpy()[0])
            # env.render()

            state = torch.tensor(prepro(state))
            epr += reward
            reward = np.clip(reward, -1, 1)
            done = done or episode_length >= args.max_frame_episode # 每局游戏最大帧长度1e4

            info['frames'].add_(1)  # torch.tensor().add()用完不改变原值，add_()会改变原值
            num_frames = int(info['frames'].item())

            if done:
                info['episodes'] += 1
                # 以下代码在做滑动平均
                interp = 1 if info['episodes'][0] == 1 else 0.01
                info['run_epr'].mul_(1 - interp).add_(interp * epr)
                info['run_loss'].mul_(1 - interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60: # 第0个worker每分钟输出1次训练信息
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                print('time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                         .format(elapsed, info['episodes'].item(), num_frames / 1e6,
                                 info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done:
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

            values.append(value)
            logps.append(logp)
            actions.append(action)
            rewards.append(reward)

        next_value = torch.zeros(1, 1) if done else model((state.unsqueeze(0), hx))[0] # if done tensor([[0.]])
        values.append(next_value.detach())

        loss = loss_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        shared_optimizer.step() #在主模型上更新梯度
    print(rank,'finish')


if __name__ == "__main__":
    if sys.version_info[0] > 2: # 判断python版本 3.X
        mp.set_start_method('spawn', force=True) # 多进程训练
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux! Or else you get a deadlock in conv2d"

    args.num_actions = gym.make(args.env).action_space.n

    torch.manual_seed(args.seed)

    shared_model = NNPolicy(num_actions=args.num_actions).share_memory()

    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    # tensor.share_memory_()将张量放入共享空间，使多个worker都可以修改
    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}

    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=worker, args=(shared_model, shared_optimizer, rank, args, info))
        p.start()
        processes.append(p)
    for p in processes: p.join()

