from torch.autograd import Variable
from collections import deque
import torch.nn.functional as F
import torch
import time
import os

from ACmodel import ActorCritic
from envs import create_env
import constants as c


class Logger():
    def __init__(self, path, mode = 'w'):
        self.path = path
        self.mode = mode
        self.item = None    # 暂时保存log的一部分
        self.titl = "Time, num steps, FPS, episode reward, episode length"
        self.form = ''

    def title(self, titl):
        self.mode = 'w'
        self.titl = titl
        with open(self.path, self.mode) as f:
            f.write(titl + '\n')

    def format(self, form):
        for (t, f) in zip(self.titl.split(", "), form.split(", ")):
            self.form += t + " " + f + ", "
        # 删除最后一个 ","
        self.form = self.form[:-2]

    def show(self):
        print(self.item)

    def add(self, *args):
        # 生成log
        self.item = self.form %args
        with open(self.path, 'a') as f:
            f.write(str(args)[1:-1] + '\n')

    def clean(self):
        os.environment("rm %s" %path)



def test(rank, shared_model, counter):
    # 生成 logger
    logger = Logger("log.csv")
    if not c.load_model:
        logger.title("Time, num steps, FPS, episode reward, episode length" )
    logger.format("%s, %i, %.0f, %.2f, %i")

    # 不计算梯度，来降低内存损失
    with torch.no_grad():

        # 创建仿真器和模型
        torch.manual_seed(c.seed + rank)

        env = create_env(c.env_name)
        env.seed(c.seed + rank)

        model = ActorCritic(env.observation_space.shape[0], c.output_size)

        # 模型不优化
        model.eval()

        state = env.reset()
        state = torch.from_numpy(state)
        reward_sum = 0
        done = True

        start_time = time.time()

        # 避免卡死(如果在200帧内一直是负奖励，则结束)
        recent_rewards = deque(maxlen = 200)
        recent_rewards.append(1)
        episode_length = 0

        while True:
            episode_length += 1

            # 拷贝最新的模型
            if done:
                model.load_state_dict(shared_model.state_dict())
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            # 型模型中得到预测的行为
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit, dim = 1)
            action = prob.multinomial(num_samples = 1).data

            state, reward, done, _ = env.step(action[0, 0])
            done = done or episode_length >= c.max_episode_length
            reward_sum += reward

            recent_rewards.append(reward)
            if max(recent_rewards) <= 0:
                done = True
                print("test stucking")

            if done:
                logger.add( time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                            counter.value,
                            counter.value / (time.time() - start_time),
                            reward_sum, 
                            episode_length )
                print("-" * 50)
                logger.show()

                torch.save(model.state_dict(), 'weights.pkl')
                print("weights saved successfully\n")

                reward_sum = 0
                episode_length = 0
                state = env.reset()

                # 在下一次测试前休眠test_interval秒
                time.sleep(c.test_interval)

            # 将state从array转为Tensor
            state = torch.from_numpy(state)
