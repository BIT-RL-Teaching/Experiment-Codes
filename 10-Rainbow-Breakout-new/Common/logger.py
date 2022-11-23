import time
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
import datetime


class Logger:
    def __init__(self, agent, train, **config):
        self.config = config
        self.agent = agent
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.running_reward = 0
        self.running_loss = 0
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
        if self.config["do_train"] and self.config["train_from_scratch"]:
            pass
        if train:
            self.writer = SummaryWriter("Logs/" + self.log_dir)

    def log_params(self):
        for k, v in self.config.items():
            self.writer.add_text(k, str(v))

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log(self, *args):
        episode, episode_reward, loss, step, beta = args

        self.writer.add_scalar(tag="train/eps_reward",
                               scalar_value=episode_reward, global_step=episode)
        self.writer.add_scalar(tag="train/eps_loss",
                               scalar_value=loss, global_step=episode)

        # save best
        if episode_reward > self.max_episode_reward:
            self.save_weights(episode, beta, 'best')
        self.max_episode_reward = max(self.max_episode_reward, episode_reward)

        if self.running_reward == 0:
            self.running_reward = episode_reward
            self.running_loss = loss
        else:
            self.running_loss = 0.99 * self.running_loss + 0.01 * loss
            self.running_reward = 0.99 * self.running_reward + 0.01 * episode_reward

        memory = psutil.virtual_memory()
        assert self.to_gb(memory.used) < 0.98 * self.to_gb(memory.total)

        if episode%100==0:
            self.save_weights(episode, beta, 'model')

            print("EP:{}| "
                  "EP_Reward:{:.2f}| "
                  "EP_Running_Reward:{:.3f}| "
                  "Running_loss:{:.3f}| "
                  "EP_Duration:{:3.3f}| "
                  "Memory_Length:{}| "
                  "Mean_steps_time:{:.3f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Beta:{:.2f}| "
                  "Time:{}| "
                  "Step:{}".format(episode,
                                   episode_reward,
                                   self.running_reward,
                                   self.running_loss,
                                   self.duration,
                                   len(self.agent.memory),
                                   self.duration / (step / episode),
                                   self.to_gb(memory.used),
                                   self.to_gb(memory.total),
                                   beta,
                                   datetime.datetime.now().strftime("%H:%M:%S"),
                                   step
                                   ))

    # 保存模型
    def save_weights(self, episode, beta, file_name):
        if file_name == 'model':
            torch.save({"online_model_state_dict": self.agent.online_model.state_dict(),
                        "optimizer_state_dict": self.agent.optimizer.state_dict(),
                        "episode": episode,
                        "beta": beta},
                       "Logs/" + self.log_dir + "/rainbow.pth")
        elif file_name == 'best':
            torch.save({"online_model_state_dict": self.agent.online_model.state_dict(),
                        "optimizer_state_dict": self.agent.optimizer.state_dict(),
                        "episode": episode,
                        "beta": beta},
                       "Logs/" + self.log_dir + "/best1.pth")

    @staticmethod
    def load_weights():
        checkpoint = torch.load("best.pth", map_location=torch.device('cpu'))
        return checkpoint
