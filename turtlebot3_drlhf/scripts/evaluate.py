#!/usr/bin/env python3
from environment.env import DRLHFEnv
import numpy as np
import scipy.io as sio
import time
from agent.sac import SAC
from agent.ddpg import DDPG
import torch
import os   
from common.replay_buffer import ReplayBuffer
from tqdm import tqdm
import rospy
from common.settings import NUM_SCAN_SAMPLES

file_name = "/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/ckp"

def eval(agent, env):
    
    for i in range(10):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            state = np.array(state)
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            step += 1
            if step > 1000:
                break
        print(f"Episode {i} reward: {total_reward}")

def eval_1(agent, env):
    for i in range(10):
        state = env.set_env()
        done = False
        total_reward = 0
        step = 0
        while not done:
            state = np.array(state)
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            step += 1
            if step > 1000:
                break

if __name__ == '__main__':
    rospy.init_node('drl_train', anonymous=True)
    state_dim = NUM_SCAN_SAMPLES + 4
    action_dim = 2
    max_action = 1.0
    episode_record = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sac = SAC(state_dim, action_dim, max_action, device)
    ddpg = DDPG(state_dim, action_dim, max_action, device)
    sac.actor.load_state_dict(torch.load("/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/ckp/pytorch_models/actor_740.pth"))
    sac.critic.load_state_dict(torch.load("/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/ckp/pytorch_models/critic_740.pth"))
    env = DRLHFEnv()
    eval_1(sac, env)

    