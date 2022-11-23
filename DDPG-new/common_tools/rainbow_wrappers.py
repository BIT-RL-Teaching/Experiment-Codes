import gym
import numpy as np

def make_atari_rainbow_diy_only(env_id):
    main_env = gym.make(env_id)
    assert 'NoFrameskip' in main_env.spec.id
    env = NoopResetEnv(main_env)
    env = RepeatActionEnv(env)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in main_env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    return env

# 自定义的一种不执行动作的环境
class NoopResetEnv:
    def __init__(self, env):
        self.noop_max = 30
        self.noop_action = 0
        self.env = env
        self.unwrapped = self.env.unwrapped
        self.observation_space = env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        self.ale = self.env.ale
        assert self.env.unwrapped.get_action_meanings()[0] == 'NOOP'
        self.observation_space = self.env.observation_space

    def reset(self):
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0

        state = None
        for _ in range(noops):
            state, _, done, _ = self.env.step(self.noop_action)
            if done:
                state = self.env.reset()

        return state

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

# 自定义的一种重复执行动作的环境
class RepeatActionEnv:
    def __init__(self, env):
        self.env = env
        self.unwrapped = self.env.unwrapped
        self.observation_space = env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        self.ale = self.env.ale
        self.successive_frame = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

# 自定义的一种带有生存周期的环境
class EpisodicLifeEnv:
    def __init__(self, env):
        self.env = env
        self.ale = self.env.ale
        self.unwrapped = self.env.unwrapped
        self.observation_space = env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        self.natural_done = True
        self.lives = 0

        self.ALELIVE_STR = "ale.lives"

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.natural_done = done
        # print('---------info = ', info)

        try:
            info[self.ALELIVE_STR]
        except KeyError:
            print('----------yes!')
            self.ALELIVE_STR = "lives"

        if self.lives > info[self.ALELIVE_STR] > 0:
            done = True
        self.lives = info[self.ALELIVE_STR]


        return state, reward, done, info

    def reset(self):
        if self.natural_done:
            state = self.env.reset()
        else:
            state, _, _, _ = self.env.step(0)
        self.lives = self.env.ale.lives()
        return state

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

# 自定义的一种可以执行一次两步的环境
class FireResetEnv:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.ale = self.env.ale
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        state, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        state, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return state

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)