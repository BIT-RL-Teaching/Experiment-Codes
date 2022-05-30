import gym
import random
import matplotlib.pyplot as plt

plt.ion()
plt.figure(1)
count_list=[]
r_list=[]

# 创建一个出租车问题的环境
env = gym.make('Taxi-v3')
# 设置学习率
alpha = 0.5
# 设置折扣因子
gamma = 0.9
# 设置贪心策略的ε，以0.05的概率随机选择动作
epsilon = 0.05

# 初始化Q表
Q = {}
# 此环境有500种离散的不同状态，有6种不同的行为，因此Q表大小为500*6
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        # Q表中的值初始值均为零
        Q[(s, a)] = 0.0


# ε-贪婪策略
def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        # 指将range(env.action_space.n)转化为list形式后，选取list中的最大的那个值，
        # 选取的根据是比较list元素里的每一个元素对应的Q值的大小
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])

# 训练
for i in range(1000):
    count_list.append(i)
    # 总reward初始化为0
    r = 0
    # env.reset()用于重置环境，回到游戏的初始状态
    state = env.reset()
    # 先随机选取一个动作
    action = epsilon_greedy_policy(state, epsilon)
    while True:
        # 渲染图像，用于输出
        env.render()

        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             1 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        #1 执行动作得到新的状态，收益，done的值为True或False，代表本轮游戏是否到了终止状态
        nextstate, reward, done, _ = env.step(action)

        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             2 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        #2 选取下一个动作
        nextaction = epsilon_greedy_policy(nextstate, epsilon)

        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             3 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        #3 更新Q表
        Q[(state, action)] += alpha * (reward + gamma * Q[(nextstate, nextaction)] - Q[(state, action)])

        # 更新动作和状态
        # SARSA的特点在这里体现，Q-learning每一步都采用ε-贪婪策略重新选取动作，
        # 而SARSA在下一步执行的动作是上一步更新Q表时采用ε-贪婪策略选取的那个动作
        # On-policy（同）: 是指产生数据的策略与评估和要改善的策略是同一个策略
        action = nextaction
        #更新状态
        state = nextstate
        # 更新回报
        r += reward
        #如果到了终止状态则结束本轮
        if done:
            break
    print("[Episode %d] Total reward: %d" % (i + 1, r))
    r_list.append(r)
    plt.plot(count_list, r_list, c='r', ls='-', marker='o', mec='b', mfc='w')  ## 保存历史数据
    plt.pause(0.01)
env.close()