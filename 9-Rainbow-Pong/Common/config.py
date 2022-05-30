#argsparse是python的命令行解析的标准模块，内置于python，不需要安装。
#这个库可以让我们值接在命令行中就可以向程序中传入参数并让程序运行。
import argparse

def get_params():
    #使用argparse的第一步是创建一个ArgumentParser对象。
    #ArgumentParser对象包含将命令行解析成Python数据类型所需的全部信息。
    #给一个ArgumentParser添加程序参数信息是通过调用add_ argument()方法完成的。
    parser = argparse.ArgumentParser(
        # 添加一个说明
        description="Variable parameters based on the configuration of the machine or user's choice")
    # 设置算法名称
    parser.add_argument("--algo", default="rainbow", type=str,
                        help="The algorithm which is used to train the agent.")
    # 设置经验池大小
    parser.add_argument("--mem_size", default=150000, type=int, help="The memory size.")
    # 设置仿真环境名称
    parser.add_argument("--env_name", default="PongNoFrameskip-v4", type=str, help="Name of the environment.")
    parser.add_argument("--interval", default=100, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    # 执行训练
    parser.add_argument("--do_train",default='true', action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--train_from_scratch", action="store_false",
                        help="The flag determines whether to train from scratch or continue previous tries.")

    parser.add_argument("--do_intro_env", action="store_true",
                        help="Only introduce the environment then close the program.")
    parser_params = parser.parse_args()
    assert parser_params.algo is not None

    #  基于Rainbow论文的默认参数
    default_params = {"lr": 6.25e-5,
                      "n_step": 3,
                      "batch_size": 32,
                      "state_shape": (84, 84, 4),
                      "max_steps": int(10e7),
                      "gamma": 0.99,
                      "tau": 0.001,
                      "train_period": 4,
                      "v_min": -10,
                      "v_max": 10,
                      "n_atoms": 51,
                      "adam_eps": 1.5e-4,
                      "alpha": 0.5,
                      "beta": 0.4,
                      "clip_grad_norm": 10.0,
                      "final_annealing_beta_steps": int(1e+6),
                      "initial_mem_size_to_train": 1000
                      }
    total_params = {**vars(parser_params), **default_params}
    print("params:", total_params)
    return total_params
