# Experiment-Codes
Code for all experiments.<br>
21 basic experiments and 4 advanced experiments.
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

## :computer: Training

We provide complete training codes.<br>
You could adapt it to your own needs.

Training
	```
	python train.py 
	```
	<br>
### Especially: <br>
For &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;COMA : 
	```
	python main.py --config=coma --env-config=sc2 with env_args.map_name=3m
	```
	<br>
For OW QMIX : 
	```
	python main.py --config=ow_qmix --env-config=overcooked with env_args.map_name=A
	```
	<br>
For &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;QMIX : 
	```
	python main.py --config=qmix --env-config=GRF
	```
	

## :checkered_flag: Testing
Testing
	```
	python test.py 
	```


## :e-mail: Contact

If you have any question, please email `shugangli@bit.edu.cn`.
