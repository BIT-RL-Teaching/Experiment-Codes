import cv2
import gym
import numpy as np
from gym.spaces.box import Box

def create_env(env_id):
    env = gym.make(env_id)
    env = Rescale42x42(env)
    env = NormalizedEnv(env)
    env.step = warpstep(env.step)
    return env

action_dict = [[-1, 0, 0], [1, 0, 0], 
            [0, 1, 0], [0, 0, 0.8], 
            [1, 0.5, 0], [-1, 0.5, 0], 
            [1, 0, 0.4], [-1, 0, 0.4],
            [0, 0, 0]]

def warpstep(step):
    def newstep(a):
        action = action_dict[int(a)]        
        return step(action)
    return newstep


# 图像的预处理
def _process_frame42(frame):
    frame = frame[:84,6:-6]
    frame = cv2.resize(frame, (42, 42))    # 放缩
    frame = frame.mean(2, keepdims=True)   # 求均值
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)                 # 值归一化
    frame = np.moveaxis(frame, -1, 0)      # 数组的某些轴移到新的位置
    return frame

# 图像放缩
class Rescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(Rescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def observation(self, observation):
        return _process_frame42(observation)

# 计算图像的均值方差并进行归一化处理
class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
