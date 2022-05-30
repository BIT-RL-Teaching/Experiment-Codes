import torch.multiprocessing as mp
import torch
import os

from ACmodel import ActorCritic
from envs import create_env
from train import train
from test import test
import constants as c
import ACoptim

def part_pretrain(model, filename = "a3c.pkl"):
    # -------------------------------------------------------------------------------#
    # -------------------------------------------------------------------------------#
    # ---------------------             1 填空           -----------------------------#
    # -------------------------------------------------------------------------------#
    # -------------------------------------------------------------------------------#
    #1 加载预训练模型
    pretrained_dict = torch.load(filename,map_location=torch.device('cpu')) 

    model_dict = model.state_dict()
    # 加载有用的参数
    pretrained_dict['actor_linear.weight'] = model_dict['actor_linear.weight']
    pretrained_dict['actor_linear.bias'] = model_dict['actor_linear.bias']
    # 加载模型
    model.load_state_dict(pretrained_dict) 


if __name__ == '__main__':

    # 设定线程和cpu
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(c.seed)
    env = create_env(c.env_name)
    # 创建仿真智能体并分享模型
    shared_model = ActorCritic(env.observation_space.shape[0], c.output_size)
    shared_model.share_memory()

    part_pretrain(shared_model)

    # 如果需要加载模型
    if c.load_model:
        shared_model.load_state_dict(torch.load("weights.pkl"),map_location=torch.device('cpu'))
        print("weights loaded sucessfully")

    # 为模型共享创建优化器
    optimizer = ACoptim.SharedAdam(shared_model.parameters(), lr = c.lr)
    optimizer.share_memory()

    processes = []
    mp.set_start_method("spawn")
    counter = mp.Value('i', 0)  # 'i' means int
    lock = mp.Lock()

    # 将测试函数放入多线程中
    p = mp.Process(target = test, args = (c.num_processes, shared_model, counter))
    p.start()
    processes.append(p)

    # 将训练函数放入多线程中
    for rank in range(0, c.num_processes):
        p = mp.Process(target = train, args = (rank, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)

    # 等待所有线程结束
    for p in processes:
        p.join()