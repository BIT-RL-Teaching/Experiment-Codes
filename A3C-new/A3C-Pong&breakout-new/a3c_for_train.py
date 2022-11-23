# Built on Sam Greydanus' Baby Advantage Actor-Critic network

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
# from scipy.misc import imresize  # preserves single-pixel info _unlike_ img = img[::2,::2]
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
from datetime import datetime

os.environ['OMP_NUM_THREADS'] = '1'


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='PongNoFrameskip-v4', type=str, help='gym environment')  # BreakoutDeterministic-v4
    parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')

    # yyx add
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_steps', default=30e6, type=float)
    parser.add_argument('--exp_postfix', default='', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load_dir', type=str)
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner
# prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.
# yyx change
prepro = lambda img: np.array(Image.fromarray(img[35:195].mean(2)).resize((80, 80))).astype(np.float32).reshape(1, 80, 80) / 255.


def printlog(args, s, end='\n', mode='a'):
    print(s, end=end)
    f = open(args.save_dir + 'log.txt', mode)
    f.write(s + '\n')
    f.close()


class NNPolicy(nn.Module):  # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # self.gru = nn.GRUCell(32 * 5 * 5, memsize)  # 使用rnn时的gru层，为简便起见不使用
        self.linear = nn.Linear(32 * 5 * 5, memsize)
        # actor和critic网络共享前层，在最后一层分别用actor_linear和critic_linear得到动作概率分布和价值估计
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.linear(x.view(-1, 32 * 5 * 5)))
        # hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(x), self.actor_linear(x)

    def try_load(self, load_dir):
        if load_dir is None:  # train from scratch
            step = 0
            return step
        paths = glob.glob(load_dir + '\\*.tar')
        assert len(paths) > 0, 'load_dir must be valid!'
        ckpts = [int(s.split('.')[-2]) for s in paths]
        ix = np.argmax(ckpts)
        step = ckpts[ix]
        self.load_state_dict(torch.load(paths[ix]))

        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        # print('step: {}'.format(step))
        return step


class SharedAdam(torch.optim.Adam):  # extend a pytorch optimizer so it shares grads across processes
    # 共享优化器，优化器中的参数将在多进程中共享
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        # self.param_groups是一个列表，每个元素是一个字典，键为dict_keys(['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'])
        for group in self.param_groups:
            # group['params']是包含各层参数对象的列表，因此p是一层参数
            for p in group['params']:
                state = self.state[p]
                # 开启共享内存
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

        # def step(self, closure=None):
        #     print('执行SharedAdam.step方法！')
        #     for group in self.param_groups:
        #         for p in group['params']:
        #             if p.grad is None: continue
        #             self.state[p]['shared_steps'] += 1
        #             self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1  # a "step += 1"  comes later
        #     super.step(closure)


def cost_func(args, values, logps, actions, rewards):
    # TODO ok
    # 使用GAE方法计算优势函数
    np_values = values.view(-1).data.numpy()
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    # actor的loss
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

    # 使用L2代价函数的方式计算critic的loss
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()
    # 最大化动作熵，鼓励探索
    entropy_loss = (-logps * torch.exp(logps)).sum()
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def train(shared_model, shared_optimizer, rank, args, info):
    # 每个进程创建一个自己的环境，用于让局部模型与该环境交互
    env = gym.make(args.env_name)
    # 每个进程初始化局部（非共享）的模型
    model = NNPolicy(channels=4, memsize=args.hidden, num_actions=args.num_actions)
    # prepro函数对状态进行预处理，例如灰度化等
    init_frame = prepro(env.reset())
    state_np = np.concatenate((init_frame, init_frame, init_frame, init_frame), axis=0)
    state = torch.tensor(state_np)

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True
    if args.eval:
        act_same_count = 0
        all_count = 0
    # 各个线程的worker共同探索80M个时间步
    while info['steps'][0] <= int(args.max_steps) or args.eval:
        # 每次与环境交互T_horizon步前，先把全局模型（主模型）的参数拷贝过来
        model.load_state_dict(shared_model.state_dict())  # 导入主模型参数
        # hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], []  # 存储经验，用于计算梯度

        for step in range(args.rnn_steps):  # 收集若干个时间步的经验
            # if args.debug and step % 100 == 0: print('step = ', step)
            if args.eval and step % 100 == 0 and all_count != 0:
                print('yyx: act same ratio = ', act_same_count / all_count)
            episode_length += 1
            value, logit = model(inputs=state.view(1, 4, 80, 80))  # 这里做了帧堆叠，是针对atari环境的常见技巧
            logp = F.log_softmax(logit, dim=-1)  # log_softmax函数等价于做完softmax操作再取对数

            # logp.max(1)[1].data if args.eval else
            # TODO eval's logic should be added? 不过不加就已经能eval21分了...
            action = logp.max(1)[1].data if args.eval else \
                torch.exp(logp).multinomial(num_samples=1).data[0]
            if args.eval:
                # print('a = ', action)
                if logp.max(1)[1].data == torch.exp(logp).multinomial(num_samples=1).data[0]:
                    act_same_count += 1
                all_count += 1
            new_state, reward, done, _ = env.step(action.numpy()[0])
            if args.eval: env.render()
            state_np = np.concatenate((prepro(new_state), state_np[:-1, :, :]), axis=0)
            state = torch.tensor(state_np)

            epr += reward
            reward = np.clip(reward, -1, 1)  # 这里手动实现了ClipReward Wrapper，也是针对Atari环境的常用技巧
            done = done or episode_length >= 1e4  # 如果与一个回合交互了太久，终止它

            info['steps'].add_(1)  # 其实就是+=1
            num_frames = int(info['steps'].item())
            if num_frames % 2e6 == 0:  # 每2M个时间步保存一次模型
                printlog(args, '\n\t{:.0f}M steps: saved model\n'.format(num_frames / 1e6))
                torch.save(shared_model.state_dict(), args.save_dir + 'model_sam_BreakoutDeterministic-v4_conv.{:.0f}.tar'.format(num_frames / 1e6))

            if done:
                # if args.debug: print('yyx: end an episode!')
                info['num_episodes'] += 1
                # horizon=0.99，含义是对最后的100个episode的reward、loss做滑动平均
                interp = 1 if info['num_episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1 - interp).add_(interp * epr)
                info['run_loss'].mul_(1 - interp).add_(interp * eploss)

            # 每分钟打印一下日志
            if rank == 0 and time.time() - last_disp_time > 60:
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, steps {:.1f} M, mean_epr {:.2f}, run_loss {:.2f}'
                         .format(elapsed, info['num_episodes'].item(), num_frames / 1e6,
                                 info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            # 重置环境
            if done:
                episode_length, epr, eploss = 0, 0, 0
                init_frame = prepro(env.reset())
                state_np = np.concatenate((init_frame, init_frame, init_frame, init_frame), axis=0)
                state = torch.tensor(state_np)

            values.append(value)
            logps.append(logp)
            actions.append(action)
            rewards.append(reward)

        next_value = torch.zeros(1, 1) if done else model(inputs=state.unsqueeze(0))[0]
        values.append(next_value.detach())
        # print(values.shape, logps.shape, actions.shape)
        # 计算actor和critic的损失函数
        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        # 清空梯度
        shared_optimizer.zero_grad()  # TODO yyx猜想，这句话可以往后挪，甚至不用加，因为后面会把model的参数拷贝给shared_model
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        # 将局部模型的梯度传给主模型
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad
        # 在主模型上，根据梯度进行更新
        shared_optimizer.step()


if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "When use python2.x, Must be using Python 3 with linux!"  # or else you get a deadlock in conv2d

    args = get_args()
    args.save_dir = 'runs/debug' if args.debug else 'runs'
    args.save_dir += f'/{args.env_name}/{datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")}'
    if args.exp_postfix != '': args.save_dir += f'_{args.exp_postfix}'
    args.save_dir += '/'

    if args.debug:
        args.processes = 2
        args.max_steps = 1e4
    if args.eval:  args.processes = 1; args.lr = 0  # eval with one process; don't train in eval mode
    args.num_actions = gym.make(args.env_name).action_space.n  # get the action space of this game
    os.makedirs(args.save_dir) if not args.eval and not os.path.exists(args.save_dir) else None  # make dir to save models etc.

    torch.manual_seed(args.seed)
    # 注意网络的api是share_memory()，tensor的api是share_memory_()
    # 在主进程中初始化一个全局模型（主模型）
    shared_model = NNPolicy(channels=4, memsize=args.hidden, num_actions=args.num_actions).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    # 这里把info中的tensor设为share_memory_()是很关键的，使得变量能在各个进程之间共享
    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'num_episodes', 'steps']}
    info['steps'] += shared_model.try_load(args.load_dir) * 1e6
    if int(info['steps'].item()) == 0: printlog(args, '', end='', mode='w')  # clear log file

    # 创建若干个worker进程并开启训练
    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start()
        processes.append(p)
    for p in processes: p.join()
