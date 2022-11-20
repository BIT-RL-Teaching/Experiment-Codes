import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, seed, device):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)  # 环境随机数种子
        # 卷积层与线性层设置
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(1 * 32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_size)
        self.device = device

    def forward(self, state):
        x = state.clone().to(self.device)  # 复制tensor并传到计算设备上
        x = x.view(-1, 4, 84, 84)  # 调整维度顺序，使通道维度在第1维，原本在最后一维

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 9 * 9)  # 将2维图像展平为1维向量
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # 输出环境中的各个动作对应的Q值
