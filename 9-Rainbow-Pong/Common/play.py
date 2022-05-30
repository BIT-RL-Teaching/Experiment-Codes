from Common.utils import *
import os


class Play:
    def __init__(self, agent, env, weights, **config):
        self.config = config
        self.agent = agent
        self.weights = weights
        self.agent.ready_to_play(self.weights)
        self.env = env
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Results"):
            os.mkdir("Results")

    #测试
    def evaluate(self):
        stacked_states = np.zeros(shape=self.config["state_shape"], dtype=np.uint8)
        total_reward = 0
        print("--------Play mode--------")
        for _ in range(1):
            done = 0
            state = self.env.reset()
            episode_reward = 0
            stacked_states = stack_states(stacked_states, state, True)
            while not done:
                stacked_frames_copy = stacked_states.copy()
                #模型产生动作
                action = self.agent.choose_action(stacked_frames_copy)
                #执行动作
                next_state, r, done, _ = self.env.step(action)
                stacked_states = stack_states(stacked_states, next_state, False)
                self.env.render()
                #更新reward的总和
                episode_reward += r
            total_reward += episode_reward

        print("Total episode reward:", total_reward)
        self.env.close()
        cv2.destroyAllWindows()
