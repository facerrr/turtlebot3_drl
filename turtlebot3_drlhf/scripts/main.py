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
from std_msgs.msg import Int32

file_name = "/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/ckp"
if not os.path.exists(file_name+"/results"):
    os.makedirs(file_name+"/results")
if not os.path.exists(file_name+"/pytorch_models"):
    os.makedirs(file_name+"/pytorch_models")

Stop = False
Save = False

def controller_callback(msg):
    global Stop, Save
    signal = msg.data
    if signal == 99:
        Stop = True
        rospy.loginfo("Get Stop Signal")
    elif signal == 1:
        Save = True
        rospy.loginfo("Get Save Signal")


def read_human_data():
    data = sio.loadmat('/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/records/records_empty_world.mat')
    states = data['states']
    actions = data['actions']
    rewards = data['rewards'][0]
    next_states = data['next_states']
    dones = data['dones'][0]
    size = len(states)
    human_buffer = ReplayBuffer(1000000)

    for i in range(len(states)):
        if actions[i][0] == 0.0 and actions[i][1] == 0.0:
            continue
        human_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    return human_buffer

def merge_buffer(replay_buffer, human_data):
    for experience in human_data.buffer:
        state, action, reward, next_state, done = experience
        replay_buffer.push(state, action, reward, next_state, done)

    return replay_buffer

def save_model(agent, episode):
    torch.save(agent.actor.state_dict(), f"{file_name}/pytorch_models/actor_{episode}.pth")
    torch.save(agent.critic.state_dict(), f"{file_name}/pytorch_models/critic_{episode}.pth")

def read_model(agent, model_file):
    agent.actor.load_state_dict(torch.load(model_file))
    agent.critic.load_state_dict(torch.load(model_file))
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic_target.load_state_dict(agent.critic.state_dict())

def save_buffer(replay_buffer,episode):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for experience in replay_buffer.buffer:
            states.append(experience[0])
            actions.append(experience[1])
            rewards.append(experience[2])
            next_states.append(experience[3])
            dones.append(experience[4])

        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}
        filename = '/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/records/buffer_{}.mat'.format(episode)
        rospy.loginfo(f"Saving buffer to {filename}")
        sio.savemat(filename, data)

def read_buffer():
    data = sio.loadmat('/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/records/buffer_917.mat')
    states = data['states']
    actions = data['actions']
    rewards = data['rewards'][0]
    next_states = data['next_states']
    dones = data['dones'][0]
    size = len(states)
    buffer = ReplayBuffer(1000000)

    for i in range(len(states)):
        if actions[i][0] == 0.0 and actions[i][1] == 0.0:
            continue
        buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    return buffer

def train(agent, human_data,replay_buffer ,start_episode=0, merge=False, num_episode=1000):
    global Stop, Save  
    NUM_STEP = 10000
    RANDOM_STEPS = 25000
    reward_buffer = []
    bar = tqdm(total=num_episode,initial=start_episode)
    total_steps = 0
    noise = 0.1*max_action
    total_episodes = 0

    if merge and start_episode == 0:
        replay_buffer = merge_buffer(replay_buffer, human_data)

    for episode in range(start_episode, num_episode):
        state = env.reset()
        total_reward = 0
        done = False
        episode_steps = 0
        last_reward = 0

        if Stop:
            break

        if Save:
            save_model(agent, total_episodes)
            save_buffer(replay_buffer, total_episodes)
            Save = False

        for step in range(NUM_STEP):
            if total_steps < RANDOM_STEPS and start_episode == 0:
                action = [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)] * action_dim

            else:
                state = np.array(state)
                if agent.__name__() == "SAC":
                    action = agent.select_action(state, deterministic=False)
                elif agent.__name__() == "DDPG":
                    action = agent.select_action(state)

                action = agent.select_action(state)
                action = action + np.random.normal(0, noise, action_dim)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done) 
            state = next_state
            total_reward += reward
            total_steps += 1
            episode_steps += 1
            last_reward = reward

            if merge:
                agent.update(replay_buffer, None)
            else:
                agent.update(replay_buffer, human_data)

            if done:
                break
        total_episodes += 1
        bar.set_description(f"Episode: {episode + 1}, Everage Reward: {total_reward/episode_steps:.2f}, Last Reward: {last_reward:.2f}")
        bar.update()
        reward_buffer.append(total_reward)

    bar.close()
    timestamp = time.strftime("%Y%m%d%H%M%S")
    torch.save(sac.actor.state_dict(), f"{file_name}/pytorch_models/actor_{timestamp}_{total_episodes}.pth")
    torch.save(sac.critic.state_dict(), f"{file_name}/pytorch_models/critic_{timestamp}_{total_episodes}.pth")
    save_buffer(replay_buffer,total_episodes)

if __name__ == '__main__':
    rospy.init_node('drl_train', anonymous=True)
    sub = rospy.Subscriber('stop_signal', Int32, controller_callback)
    state_dim = NUM_SCAN_SAMPLES + 4
    action_dim = 2
    max_action = 1.0
    episode_record = 2000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sac = SAC(state_dim, action_dim, max_action, device)
    ddpg = DDPG(state_dim, action_dim, max_action, device)
    # replay_buffer = ReplayBuffer(1000000)
    replay_buffer = read_buffer()
    sac.actor.load_state_dict(torch.load("/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/ckp/pytorch_models/actor_2000.pth"))
    sac.critic.load_state_dict(torch.load("/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/ckp/pytorch_models/critic_2000.pth"))
    sac.critic_target.load_state_dict(sac.critic.state_dict())
    env = DRLHFEnv()
    human_data = read_human_data()
    train(sac, replay_buffer, human_data, episode_record, merge=True, num_episode=2500)
    # rospy.on_shutdown(save_model(sac, episode_record))
    # rospy.spin()
    