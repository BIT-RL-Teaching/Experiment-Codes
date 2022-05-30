import os
import gym
import shutil
import argparse
import numpy as np
from tqdm import trange
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 一种缓冲区，用于存储PPO代理与环境交互所经历的轨迹，
# 并使用广义优势估计（GAE-Lambda）计算状态-动作对的优势。
# 创建ppo经验池
class PPOMemory:
    def __init__(self, gamma, tau):
        self.states = []        # 环境信息
        self.actions = []       # 动作信息
        self.rewards = []       # 奖励信息
        self.values = []        # 值函数
        self.logprobs = []      # 对数概率
        self.tdlamret = []      # advants + values
        self.advants = []       # 优势函数值
        self.gamma = gamma      # gamma-just的gamma参数
        self.tau = tau          # τ是用于计算gae的一个参数
        self.ptr = 0                #数组最后一个位置的索引
        self.path_start_idx = 0     #数组开始位置的索引

    # 存储数据
    def store(self, s, a, r, v, lp):
    	# append() 方法用于在列表末尾添加新的对象。
    	# 该方法无返回值，但是会修改原来的列表。
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.values.append(v)
        self.logprobs.append(lp)
        self.ptr += 1

    # 当结束了一个路径后
    def finish_path(self, v):
        # 制作一批数据
        # np.arange函数返回一个有终点和起点的固定步长的排列，
        # 两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1
        path_slice = np.arange(self.path_start_idx, self.ptr)
        # list 转 numpy
        rewards_np = np.array(self.rewards)[path_slice]
        values_np = np.array(self.values)[path_slice]
        values_np_added = np.append(values_np, v)
        # 初始化gae=0
        gae = 0
        advants = []
        # 这里使用了A3C算法中的N步回报价值估计法
        for t in reversed(range(len(rewards_np))):
        	# delta的计算采用TD-error的形式
        	# 公式通过当前时刻的回报和下一时刻的价值估计得到了目标价值
        	# 然后减去当前时刻的价值估计
            delta = rewards_np[t] + self.gamma * values_np_added[t+1] - values_np_added[t]
            # 利用gae的累加公式来计算出优势函数的估计值
            gae = delta + self.gamma * self.tau * gae
            # python list的insert()函数用于将指定对象插入列表的指定位置。
            advants.insert(0, gae)
        # python list的extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
        self.advants.extend(advants)
        # list 转 numpy
        advants_np = np.array(advants)
        # 将优势函数和值函数对应位置相加求和
        tdlamret_np = advants_np + values_np
        # python list的extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
        self.tdlamret.extend(tdlamret_np.tolist())
        self.path_start_idx = self.ptr

    def reset_storage(self):
    	# 重置数值
        self.ptr, self.path_start_idx = 0, 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.tdlamret = []
        self.advants = []

    def get(self):
        # 返回数据用于更新模型  
        # python dict() 函数用于创建一个字典。k=states -- v==self.states
        data = dict(states=self.states, actions=self.actions, logpas=self.logprobs,
                    rewards=self.rewards, values=self.values,
                    tdlamret=self.tdlamret, advants=self.advants)
        # ppo是on-policy策略，ppo在计算梯度时，是通过目标函数的梯度进行策略更新
        # 所以每一次计算梯度时，需要使用当前最新的策略模型重新进行交互采样，得到相应的序列样本，
        # 然后使用这些样本完成梯度的计算
        # 数据使用后就立刻清空
        self.reset_storage()
        return data
    # 返回经验池的长度
    def __len__(self):
        return len(self.rewards)

class PPO():
    def __init__(self):
        super(PPO, self).__init__()
        self.seed = 66                      # 固定随机种子
        self.average_interval = 100         # 优势间隔
        self.gae_tau = 0.95                 # 原论文给的合适的参数
        self.gamma = 0.99                   # gamma-just的gamma参数
        self.max_episodes = 5000            # 最大的片段数量
        self.max_steps_per_episode = 300    # 每个片段执行多少步
        self.batch_size = 32                # 一批数据的数量
        self.clip_range = 0.2               # CLIP参数
        self.coef_entpen = 0.001            # 目标函数L中的熵S的参数c2
        self.coef_vf = 0.5                  # 目标函数L中的VF的参数c1
        self.memory_size = 2048             # 经验池的最大容量
        self.optim_epochs = 4               # 优化的论述
        self.terminal_score = 230           # 最终目标得分
        self.evn_name = "LunarLander-v2"    # 环境的名称
        self.lr = 0.002                     # 学习率的值
        self.betas = [0.9,0.999]            # 用于初始化pytorch的Adam优化器
        self.game = gym.make(self.evn_name) # 仿真环境
        self.input_dim = self.game.observation_space.shape[0]  # 输入数据维度
        self.output_dim = self.game.action_space.n             # 输出数据维度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 为CPU设置种子用于生成随机数，以使得结果是确定的
        torch.manual_seed(self.seed)
        # 利用随机数种子，使得numpy每次生成的随机数相同。
        np.random.seed(self.seed)
        # 创建actor对象
        self.actor = Actor(device=self.device,input_dim=self.input_dim, output_dim=self.output_dim,)
        # 创建actor优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.lr,betas=self.betas)
        self.critic = Critic(device=self.device , input_dim=self.input_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.lr,betas=self.betas)
    @staticmethod
    def load_weight(model):
        #读取模型，此处为静态方法，原因在于在本PPO类中会调用，只是类实例化之前的静态变量和函数等问题和RL无关
        weight = torch.load("ppo.pth")
        model.load_state_dict(weight)
    @staticmethod
    def save_weight(model):
        #保存模型，可以不实例化调用该方法
        weight = model.state_dict()
        torch.save(weight, "ppo.pth")
    # 训练
    def train(self):
        # 设置 env, memory, stuff
        env = gym.make(self.evn_name)
        # 使用固定的种子
        env.seed(self.seed)
        # 创建ppo经验池对象
        self.memory = PPOMemory(gamma=self.gamma , tau=self.gae_tau)
        # 建立得分队列
        score_queue = deque(maxlen=self.average_interval)
        # 建立长度队列
        length_queue = deque(maxlen=self.average_interval)
        # trange(i) 是 tqdm(range(i)) 的等价写法
        for episode in trange(1, self.max_episodes+1):
            self.episode = episode
            episode_score = 0
            # 重置环境 获得state
            state = env.reset()
            for t in range(1, self.max_steps_per_episode+1):
                # -------------------------------------------------------------------------------#
                # -------------------------------------------------------------------------------#
                # ---------------------             1 填空           -----------------------------#
                # -------------------------------------------------------------------------------#
                # -------------------------------------------------------------------------------#
                #每隔100个episode渲染一次环境输出情况
                if self.episode % 100 == 0:
                    env.render()
                with torch.no_grad():
                	# 对输入的指定位置插入维度 1，并返回一个新的张量，
                	# 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。
                    state_tensor = torch.tensor(state).unsqueeze(0).float()
                    # actor选择行为以及计算其对应的对数概率
                    action_tensor, logpa_tensor = self.actor.select_action(state_tensor)
                    # critic计算价值函数
                    value_tensor = self.critic(state_tensor).squeeze(1)
                # 执行动作
                action = action_tensor.numpy()[0] 
                # 获得返回的信息
                next_state, reward, done, _ = env.step(action)
                # 更新 episode_score
                episode_score += reward
                # -------------------------------------------------------------------------------#
                # -------------------------------------------------------------------------------#
                # ---------------------             2 填空           -----------------------------#
                # -------------------------------------------------------------------------------#
                # -------------------------------------------------------------------------------#
                # 向经验池中添加经验  一个元素张量可以用x.item()得到元素值
                self.memory.store(s=state, a=action, r=reward, v=value_tensor.item(), lp=logpa_tensor.item())
                # 判断是否达到了最大的步数
                timeout = t == self.max_steps_per_episode
                # 判断是否该进行优化
                time_to_optimize = len(self.memory) == self.memory_size
                # 如果游戏结束 或 超市 或 经验池满了
                if done or timeout or time_to_optimize:
                    if done:
                        # 因为游戏结束，下一个状态的值为0
                        v = 0
                    else:
                        # 如果没结束，则与critic一起估算价值函数
                        next_state_tensor = torch.tensor(next_state).unsqueeze(0).float()
                        with torch.no_grad():
                            next_value_tensor = self.critic(next_state_tensor).squeeze(1)
                        v = next_value_tensor.item()
                    # 更新 gae & tdlamret
                    self.memory.finish_path(v)
                # 如果经验池满了, 优化 PPO
                if time_to_optimize:
                    # -------------------------------------------------------------------------------#
                    # -------------------------------------------------------------------------------#
                    # ---------------------             3 填空           -----------------------------#
                    # -------------------------------------------------------------------------------#
                    # -------------------------------------------------------------------------------#
                    #梯度的反向传播
                    self.optimize()
                # 如果结束了，就把相应的值加入到相应的队列中
                if done:
                    score_queue.append(episode_score)
                    length_queue.append(t)
                    break
                # 更新状态
                state = next_state
            # 计算平均得分
            avg_score = np.mean(score_queue)
            # 计算标准差
            std_score = np.std(score_queue)
            # 得到平均后的时间点
            avg_duration = np.mean(length_queue)
            if self.episode % 100 == 0:
                print("{} - score: {:.1f} +-{:.1f} \t duration: {}".format(self.episode, avg_score, std_score, avg_duration))
                print("found best model at episode: {}".format(self.episode))
                self.save_weight(self.actor)
        #保存模型
        self.save_weight(self.actor)
        return avg_score
    # 优化
    def optimize(self):
        data = self.prepare_data(self.memory.get())
        self.optimize_ppo(data)

    def prepare_data(self, data):
        # 从PPO-memory中获得一组数据，数据存放在data字典中，然后分别将他们拿出来并转换相应的类型
        # torch.tensor转换过程中会进行数据拷贝，不会共享内存
        states_tensor = torch.from_numpy(np.stack(data['states'])).float()
        actions_tensor = torch.tensor(data['actions']).long()
        logpas_tensor = torch.tensor(data['logpas']).float()
        tdlamret_tensor = torch.tensor(data['tdlamret']).float()
        advants_tensor = torch.tensor(data['advants']).float()
        values_tensor = torch.tensor(data['values']).float()

        # 经过加工处理数据分布会更好，有一定可能产生更好效果，但也可以不进行
        advants_tensor = (advants_tensor - advants_tensor.mean()) / (advants_tensor.std() + 1e-5)
        # 将数据写入字典并返回
        data_tensor = dict(states=states_tensor, actions=actions_tensor, logpas=logpas_tensor,
                    tdlamret=tdlamret_tensor, advants=advants_tensor, values=values_tensor)
        return data_tensor
    #建立一个循环的迭代器用于PPO训练
    def ppo_iter(self, batch_size, ob, ac, oldpas, atarg, tdlamret, vpredbefore):
        total_size = ob.size(0)
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        # 计算要返回多少次batch
        n_batches = total_size // batch_size
        for nb in range(n_batches):
            ind = indices[batch_size * nb : batch_size * (nb+1)]
            # yield 的作用就是把一个函数变成一个 generator，不再是一个普通函数，
            # Python 解释器会将带有 yield 的函数视为一个 generator，返回一个 iterable 对象
            yield ob[ind], ac[ind], oldpas[ind], atarg[ind], tdlamret[ind], vpredbefore[ind]

    def optimize_ppo(self, data):
        # 首先从经验池中获取数据
        ob = data['states']
        ac = data['actions']
        oldpas = data['logpas']
        atarg = data['advants']
        tdlamret = data['tdlamret']
        vpredbefore = data['values']
        eps = self.clip_range
        # 保存计算的损失值
        policy_losses = []
        entropy_losses = []
        value_losses = []

        # 对每个epoch
        for i in range(self.optim_epochs):
            # 对每个 batch
            data_loader = self.ppo_iter(self.batch_size,
                                        ob, ac, oldpas, atarg, tdlamret, vpredbefore)
            for batch in data_loader:
                ob_b, ac_b, old_logpas_b, atarg_b, vtarg_b, old_vpred_b = batch
                #计算新旧策略的比率，
                cur_logpas, cur_entropies = self.actor.get_predictions(ob_b, ac_b)
                # torch.exp实现计算以e为底的指数
                ratio = torch.exp(cur_logpas - old_logpas_b)
                # 限制比率，确保每次更新都不会发太大的波动
                clipped_ratio = torch.clamp(ratio, 1.-eps, 1.+eps)
                surr1 = ratio * atarg_b
                surr2 = clipped_ratio * atarg_b
                # 选择较低值进行优化，如果能将较小的值优化到令人满意的程度
                # 那么对于其他的情况，模型的表现会更好
                pol_surr = -torch.min(surr1, surr2).mean()
                cur_vpred = self.critic(ob_b).squeeze(1)
                vf_loss = (cur_vpred - vtarg_b).pow(2).mean()
                # 计算熵损失
                pol_entpen = -cur_entropies.mean()
                c1 = self.coef_vf
                c2 = self.coef_entpen
                # actor net的反向优化
                # 把梯度置零
                self.actor_optimizer.zero_grad()
                # 计算loss
                policy_loss = pol_surr + c2 * pol_entpen
                # 计算梯度
                policy_loss.backward()
                # 反向优化
                self.actor_optimizer.step()
                # critic net的反向优化
                self.critic_optimizer.zero_grad()
                value_loss = c1 * vf_loss
                value_loss.backward()
                self.critic_optimizer.step()
                # loss数组值的更新
                policy_losses.append(pol_surr.item())
                entropy_losses.append(pol_entpen.item())
                value_losses.append(vf_loss.item())

    # 模型测试
    def play(self, num_episodes=1,seed=9999):
        # 加载策略
        self.load_weight(self.actor)
        env = gym.make(self.evn_name)
        env.seed(seed)
        scores = []
        for episode in range(num_episodes):
            episode_score = 0
            # 初始化环境
            state = env.reset()
            while True:
                env.render()
                # 选择动作
                with torch.no_grad():
                    action_tensor = self.actor.select_greedy_action(state)
                action = action_tensor.numpy()[0] # single env
                # 执行 action
                next_state, reward, done, _ = env.step(action)
                # 更新 reward
                episode_score += reward
                # 更新 state
                state = next_state
                if done:
                    scores.append(episode_score)
                    break
        avg_score = np.mean(scores)
        print("RESULT: Average score {:.3f} on {} {} games".format(avg_score, num_episodes,self.evn_name))
        env.close()

#初始化
def init_normal_weights(m):
    if isinstance(m, nn.Linear):
    	# 使用均匀分布初始化
        nn.init.normal_(m.weight, mean=0., std=0.1)
        # 初始化偏执为常数
        nn.init.constant_(m.bias, 0.1)

class Actor(nn.Module):
    def __init__(self, device, input_dim, output_dim,):
        super(Actor, self).__init__()
        # 定义Actor的各个网络层
        self.input_layer = nn.Linear(input_dim, 64)
        self.hidden_layers = nn.ModuleList()
        for idx in range(1):
            self.hidden_layers.append(nn.Linear(64, 64))
        self.output_layer = nn.Linear(64, output_dim)
        self.hfn = torch.tanh
        self.apply(init_normal_weights)
        self.device = device
    # 动作的选择
    def select_action(self, states):
    	# 计算动作的概率
        probs = self.forward(states)
        # Categorical创建以参数probs为标准的类别分布
        dist = Categorical(probs=probs)
        # 随机采样动作
        actions = dist.sample()
        # 计算动作的对数概率 
        log_probs = dist.log_prob(actions)
        return actions, log_probs
    # 选择可能性最高的动作
    def select_greedy_action(self, states):
        probs = self.forward(states)
        _, actions = probs.max(1)
        return actions
    # 获取旧动作的log_probs和当前分布的熵
    def get_predictions(self, states, old_actions):
        state, old_actions = self._format(states), self._format(old_actions)
        # 计算动作的概率
        probs = self.forward(states)
        # Categorical创建以参数probs为标准的类别分布
        dist = Categorical(probs=probs)
        # 计算动作的对数概率 
        log_probs = dist.log_prob(old_actions)
        # 计算熵
        entropies = dist.entropy()
        return log_probs, entropies
    # 定义模型的前向传播
    def forward(self, state):
        state = self._format(state)
        x = self.input_layer(state)
        x = self.hfn(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.hfn(x)
        x = self.output_layer(x)
        x = torch.softmax(x, dim=1)
        return x
    # 将numpy转换为张量并添加一个维度
    def _format(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            state = state.unsqueeze(0) 
        return state

class Critic(nn.Module):
	# 定义Actor的各个网络层
    def __init__(self, device, input_dim):
        super(Critic, self).__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.hidden_layers = nn.ModuleList()
        for idx in range(1):
            self.hidden_layers.append(nn.Linear(64, 64))
        self.output_layer = nn.Linear(64, 1)
        self.hfn = torch.tanh
        self.apply(init_normal_weights)
        self.device = device
    # 定义模型的前向传播
    def forward(self, state):
        state = self._format(state)
        x = self.input_layer(state)
        x = self.hfn(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.hfn(x)
        x = self.output_layer(x)
        return x
    # 将numpy转换为张量并添加一个维度
    def _format(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = state.unsqueeze(0)  # 
        return state

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--eval",action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # 创建PPO对象
    agent = PPO()
    if args.eval:
    	# 测试模型
        agent.play(num_episodes=args.eval_episodes, seed=args.seed)
    else:
    	# 训练模型
        print("Training PPO agent on game {}...".format(agent.evn_name))
        agent.train()
        print("Done\n")
        agent.game.close()

if __name__ == "__main__":
    main()
