import numpy as np
import scipy.io as sio
import time
from replay_buffer import ReplayBuffer
import csv
def read_human_data(replay_buffer):
    data = sio.loadmat('/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/records/buffer_1.mat')
    states = data['states']
    actions = data['actions']
    rewards = data['rewards'][0]
    next_states = data['next_states']
    dones = data['dones'][0]
    print(rewards)

    for i in range(len(states)):
        if actions[i][0] == 0.0 and actions[i][1] == 0.0:
            continue
        replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    
    state, action, reward, next_state, done = replay_buffer.sample(10)
    print(state)
    print(action)
    print(reward)

    

if __name__ == '__main__':

    replay_buffer = ReplayBuffer(1000000)
    read_human_data(replay_buffer)
