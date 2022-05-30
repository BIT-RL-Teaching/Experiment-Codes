import gym
import keyboard
import time

env = gym.make('Breakout-v0')
state = env.reset()
action = 0

##############################
# 1--start  2--left  3--rihgt  
##############################
def key(x):
    global action
    if x.event_type == "down" and x.name == '2':
        action = 3
    elif x.event_type == "down" and x.name == '3':
        action = 2
    elif x.event_type == "down" and x.name == '1':
    	action = 1
    else:
    	action = 0
keyboard.hook(key)
total_reward = 0
while True:
    env.render()
    next_state,reward,done,_ = env.step(action)
    total_reward += reward
    print(total_reward)
    time.sleep(0.05)
    if done:
        env.reset()
keyboard.wait()
env.close()