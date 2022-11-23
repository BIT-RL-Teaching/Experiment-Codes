import gym
import numpy as np

class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, args):
        self.args = args
        self.env = gym.make('CarRacing-v0')
        self.env.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold

        shape = (4, ) + self.env.observation_space.shape[:2]  # 这里是(4, )而不是(1, )，并在外部取消VecPyTorchFrameStack类的包装
        self.observation_space = gym.spaces.Box(
            low=np.min(self.env.observation_space.low),
            high=np.max(self.env.observation_space.high),
            shape=shape,
            dtype=self.env.observation_space.dtype
        )
        self.action_space = self.env.action_space
        self.reward_range = (float('-inf'), float('inf'))
        self.metadata = None
        class TMP():
            def __init__(self):
                self.id = None
        self.spec = TMP()

    # def __getattr__(self, item):  # 本意是想配合yyx_make_CarRacing_env中的env.seed = seed，但暂时没搞懂
    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        # yyx: change to the tianshou usage, 'img_stack' -> 'frames_stack'
        self.stack = [img_gray] * self.args.frames_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.args.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die and not self.args.debug_dont_compensate_die_state:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 or die else False  # TODO 这里关于die和done的判断做了小修改，学不出来的话来检视
            if done:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        # yyx: change to the tianshou usage, 'img_stack' -> 'frames_stack'
        assert len(self.stack) == self.args.frames_stack
        return np.array(self.stack), total_reward, done, {}

    def render(self, *arg):
        self.env.render(*arg)

    def close(self):
        self.env.close()

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory