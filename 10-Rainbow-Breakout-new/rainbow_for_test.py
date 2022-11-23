from Common.logger import Logger
from Common.play import Play
from Brain.agent import Agent
from Common.utils import *
from Common.config import get_params
import time

if __name__ == '__main__':
    params = get_params()
    #创建测试环境
    env = make_atari(params["env_name"], episode_life=False)
    params.update({"n_actions": env.action_space.n})
    # env.seed(int(time.time()))
    #初始化智能体
    agent = Agent(**params)
    logger = Logger(agent, train=False, **params)
    #加载已经训练好的模型
    chekpoint = logger.load_weights()
    player = Play(agent, env, chekpoint["online_model_state_dict"], **params)
    #开始测试模型
    test_result = []
    for game in range(10):
        env.seed(int(time.time()))
        test_result.append(player.evaluate())
    print('avg score:{:.2f}'.format(np.mean(test_result)))
