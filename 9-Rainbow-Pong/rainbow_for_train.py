from Common.logger import Logger
from Brain.agent import Agent
from Common.utils import *
from Common.config import get_params
import time

if __name__ == '__main__':
    #参数初始化
    params = get_params()
    test_env = make_atari(params["env_name"])
    params.update({"n_actions": test_env.action_space.n})
    print(f"Environment: {params['env_name']}\n"
          f"Number of actions:{params['n_actions']}")
    # 创建训练环境
    env = make_atari(params["env_name"])
    #所有gym环境都使用了numpy的随机数生成器。如果种子不同，生成器得产生不同的随机数序列。
    #由于机器学习受经验驱动，因此可重复性非常重要。所以为了实现可重复性，我们就可以手动指定种子。
    #这里的time time()返回当前时间的时间戳，是一个随机的种子
    env.seed(int(time.time()))
    #创建智能体对象
    agent = Agent(**params)
    # logger对象用于保存实验数据和加载模型
    logger = Logger(agent, **params)
    # 使用预训练模型
    if not params["train_from_scratch"]:
        chekpoint = logger.load_weights()
        agent.online_model.load_state_dict(chekpoint["online_model_state_dict"])
        agent.hard_update_of_target_network()
        params.update({"beta": chekpoint["beta"]})
        min_episode = chekpoint["episode"]
        print("Keep training from previous run.")
    # 从头开始训练模型
    else:
        min_episode = 0
        print("Train from scratch.")

    #执行训练
    if params["do_train"]:
        #建立一个存储环境状态队列的变量，为一个样本
        #因为有的环境画面会出现闪烁等现象，此时可能会丢失某些重要信息
        #所以对连续的一定长度的画面进行存储.以保证训练数据的有效性
        stacked_states = np.zeros(shape=params["state_shape"], dtype=np.uint8)
        # 重置
        state = env.reset()
        # 将重置后的第一个环境状态加到stacked_states中
        stacked_states = stack_states(stacked_states, state, True)
        episode_reward = 0
        #beta的用于调整和权衡rainbow中优先回放的经验池中样本的重要性，
        #进而调整每个样本对模型更新的影响，即采用重要性的方法来对样本进行采样
        beta = params["beta"]
        loss = 0
        episode = min_episode + 1
        logger.on()
        for step in range(1, params["max_steps"] + 1):
            stacked_states_copy = stacked_states.copy()
            # 选择行为
            action = agent.choose_action(stacked_states_copy)
            # 执行
            next_state, reward, done, _ = env.step(action)

            # 累积状态
            stacked_states = stack_states(stacked_states, next_state, False)
            reward = np.clip(reward, -1.0, 1.0)
            # 保存
            agent.store(stacked_states_copy, action, reward, stacked_states, done)
            # 累加奖励
            episode_reward += reward

            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            # ---------------------             2 填空           -----------------------------#
            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            #2 使用多步回报，在训练的前期目标价值可以估计地更准，从而加快模型的训练
            if step % params["train_period"] == 0:
                #由于目前对于优先回放经验池这一方法的收敛性并不确定，所以研究者们还是希望
                #Priority Replay Buffer最终能变成Replay Buffer,所以VII练开始时赋值beta一个
                #小于1的数值，然后随着迭代数的增加，让beta不断地变大，并最终达到1
                #这样Priority Replay Buffer最终变成了Replay Buffer
                #同时我们既可以加快训练的速度，也可以让模型的收敛性得到保证
                beta = min(1.0, params["beta"] + step * (1.0 - params["beta"]) / params["final_annealing_beta_steps"])
                # 模型训练返回损失loss
                loss += agent.train(beta)
            # 模型的软更新
            agent.soft_update_of_target_network()

            if done:
                logger.off()
                logger.log(episode, episode_reward, loss, step, beta)
                episode += 1
                state = env.reset()
                stacked_frames = stack_states(stacked_states, state, True)
                episode_reward = 0
                episode_loss = 0
                logger.on()

