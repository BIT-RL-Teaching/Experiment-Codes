# Experiment-Codes
Code for all experiments.
21 basic experiments, 4 advanced experiments.
## :wrench: Basic Dependencies
- Python == 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch == 1.6.0](https://pytorch.org/)

### Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-RL-Teaching/Experiment-Codes.git
    cd Experiment-Codes
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```

```
## :computer: Training

We provide complete training codes for FD-MAPPO (Cubic Map).<br>
You could adapt it to your own needs.

1. If you don't have NVIDIA RTX 3090, you should comment these two lines in file
[human_drone_SC/code/util.py](https://github.com/BIT-MCS/human_drone_SC/tree/main/code/util.py).
	```
	[24]  torch.backends.cuda.matmul.allow_tf32 = False
	[25]  torch.backends.cudnn.allow_tf32 = False
	```
2. You can modify the config files 
[human_drone_SC/code/environment/KAIST/conf.py](https://github.com/BIT-MCS/human_drone_SC/tree/main/code/environment/KAIST/conf.py) and
[human_drone_SC/code/environment/NCSU/conf.py](https://github.com/BIT-MCS/human_drone_SC/tree/main/code/environment/NCSU/conf.py) for environments.<br>
For example, you can control the number of drones in the environment by modifying this line
	```
	[43]  'uav_num': 6,
	```
3. You can modify the config file 
[human_drone_SC/code/method/fd_mappo_cubicmap/conf.py](https://github.com/BIT-MCS/human_drone_SC/tree/main/code/method/fd_mappo_cubicmap/conf.py) for method.<br>
For example, you can control the hyperparameters studied in paper by modifying these two lines
	```
	[34]  'M_size': [16, 16, 16],  # Z, X, Y
	[35]  'mtx_size': 3,  # X' (Y')
	```
4. Training
	```
	python main.py KAIST fd_mappo_cubicmap train
	python main.py NCSU fd_mappo_cubicmap train
	```
	The log files will be stored in [human_drone_SC/log](https://github.com/BIT-MCS/human_drone_SC/tree/main/log).
## :checkered_flag: Testing
1. Before testing, you should modify the file [human_drone_SC/code/env_method_set.py](https://github.com/BIT-MCS/human_drone_SC/tree/main/code/env_method_set.py) to ensure the datetime of the version you want to test is right.
	```
	[2]  'KAIST/fd_mappo_cubicmap': '2021-05-27/23-48-01',
	[3]  'NCSU/fd_mappo_cubicmap': '2021-05-20/16-56-41',
	```
2. Testing
	```
	python main.py KAIST fd_mappo_cubicmap test
	python main.py NCSU fd_mappo_cubicmap test
	```


## :e-mail: Contact

If you have any question, please email `shugangli@bit.edu.cn`.
