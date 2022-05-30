from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

# 用于修剪输出，以计算全链接层的输入大小
def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - kernel_size) // stride + 1


class Model(nn.Module):
    def __init__(self, state_shape, n_actions, n_atoms, support):
        super(Model, self).__init__()
        width, height, channel = state_shape
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.n_atoms = n_atoms
        self.support = support
        # 定义网络
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        convw = conv2d_size_out(conv2d_size_out(width, kernel_size=8, stride=4), kernel_size=4, stride=2)
        convh = conv2d_size_out(conv2d_size_out(height, kernel_size=8, stride=4), kernel_size=4, stride=2)

        convw = conv2d_size_out(convw, kernel_size=3, stride=1)
        convh = conv2d_size_out(convh, kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64

        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             3 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        #3 创建一个输入为linear_input_size，输出为512的NoisyLayer
        self.adv_fc = NoisyLayer(linear_input_size, 512)
        self.adv = NoisyLayer(512, self.n_actions * self.n_atoms)

        self.value_fc = NoisyLayer(linear_input_size, 512)
        self.value = NoisyLayer(512, self.n_atoms)
        # 参数值初始化
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # 对卷积层进行凯明初始化
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                # 偏置项初始化为0
                m.bias.data.zero_()
    # 网络的前向传播
    def forward(self, inputs):
        x = inputs / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        adv_fc = F.relu(self.adv_fc(x))
        adv = self.adv(adv_fc).view(-1, self.n_actions, self.n_atoms)
        value_fc = F.relu(self.value_fc(x))
        value = self.value(value_fc).view(-1, 1, self.n_atoms)

        mass_probs = value + adv - adv.mean(1, keepdim=True)
        return F.softmax(mass_probs, dim=-1).clamp(min=1e-3)

    def get_q_value(self, x):
        dist = self(x)
        q_value = (dist * self.support).sum(-1)
        return q_value

    def reset(self):
        self.adv_fc.reset_noise()
        self.adv.reset_noise()
        self.value_fc.reset_noise()
        self.value.reset_noise()
#噪声网络的实现
#噪声网络是一种以更加平滑的方式增加模型探萦能力的方法
#我们向值函数中加入一定嗓声.嗓声会影呐.终的价值输出
#也会影响.终的行动，因此加入的嗓声会影呐模型的探萦能力
#加人的噪声越大，模型的探索能力也就越大
class NoisyLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(NoisyLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        #torch. nn. Parameter是继承自torch.下ensor的子类，其主要作用是作为nn.Module中的可训练参数使用。
        #它与torch. Tensor的区别就是nn.Parameter会自动被认为旱mndule的可训纹参数
        self.mu_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))
        self.sigma_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))
        #self.register buffer可以j tensor注册成buffer，网络存储时也会将buffer存下
        #buffer的更新在forward中，optim.step只能更新nn.parameter类型的参数。
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.n_outputs, self.n_inputs))

        self.mu_b = nn.Parameter(torch.FloatTensor(self.n_outputs))
        self.sigma_b = nn.Parameter(torch.FloatTensor(self.n_outputs))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.n_outputs))
        # uniform_实现将tensor用从均匀分布中抽样得到的值填充
        self.mu_w.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_w.data.fill_(0.5 / np.sqrt(self.n_inputs))

        self.mu_b.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_b.data.fill_(0.5 / np.sqrt(self.n_outputs))

        self.epsilon_i = 0
        self.epsilon_j = 0
        self.reset_noise()

    def forward(self, inputs):
        x = inputs
        # 添加噪声
        weights = self.mu_w + self.sigma_w * self.weight_epsilon 
        biases = self.mu_b + self.sigma_b * self.bias_epsilon
        #           输入 权重     偏置
        x = F.linear(x, weights, biases)
        return x

    def f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        self.epsilon_i = self.f(torch.randn(self.n_inputs))
        self.epsilon_j = self.f(torch.randn(self.n_outputs))
        self.weight_epsilon.copy_(self.epsilon_j.ger(self.epsilon_i))
        self.bias_epsilon.copy_(self.epsilon_j)
