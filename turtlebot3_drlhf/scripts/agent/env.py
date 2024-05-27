import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from ddpg import ReplayBuffer, DDPG
from tqdm import tqdm
import imageio
import random
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


MODEL_PATH = './run/DDPG/ckpt/'
SAVE_PATH_PREFIX = './run/DDPG/'

def evalueate_policy(env, agent, num_episodes=3):
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        rewards.append(episode_reward)
    return np.mean(rewards)

env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)
n_states = env.observation_space.shape[0]

n_actions = env.action_space.shape[0]
# n_actions = 1
# max_action = 1.0
max_action = float(env.action_space.high[0])
max_episode_steps = env._max_episode_steps

print("state_dim:", n_states)
print("action_dim:", n_actions)
print("max_action:", max_action)
print("max_episode_steps:", max_episode_steps)

replay_buffer = ReplayBuffer(capacity=100000)
agent = DDPG(state_dim=n_states, action_dim=n_actions, hidden_dim=256, max_action=max_action, sigma=0.1, tau=0.001, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, device=device)


        
def train_2():
    NUM_EPISODE = 200
    NUM_STEP = 1000
    random_steps = 10000
    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY = 10000
    RANDOM_STEPS = 64
    reward_buffer = []
    bar = tqdm(total=NUM_EPISODE)
    total_steps = 0
    noise = 0.1*max_action

    for episode_i in range(NUM_EPISODE):
        state = env.reset()
        episode_reward = 0
        done = False
        for step_i in range(NUM_STEP):
            if total_steps<random_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action = (action + np.random.normal(0, noise, size=action.shape)).clip(-max_action, max_action)

            # if total_steps < RANDOM_STEPS:
            #     action = env.action_space.sample()
            # else:
            #     action = agent.select_action(state)
            #     action = (action + np.random.normal(0, noise, size=action.shape)).clip(-max_action, max_action)

            # threshold = 0.5
            # print(action)
            # action = (action > threshold).astype(int)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            agent.learn(replay_buffer)
            if done:
                break
        bar.set_description(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")
        bar.update()
        reward_buffer.append(episode_reward)
        # print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")
    
    timestamp = time.strftime("%Y%m%d%H%M%S")
    torch.save(agent.actor.state_dict(), f"/home/face/Desktop/DRL/DDPG/ckpt/actor/{timestamp}.pth")
    torch.save(agent.critic.state_dict(), f"/home/face/Desktop/DRL/DDPG/ckpt/critic/{timestamp}.pth")

def train():
    
    # writter =SummaryWriter("runs/ddpg")
    noise = 0.1*max_action
    max_training_steps = 100000
    random_steps = 10000
    update_interval = 50
    evaluate_interval = 1000
    evaluate_num = 0
    evaluate_rewards = []
    total_steps = 0

    bar = tqdm(total=max_training_steps)

    while total_steps < max_training_steps:
        state = env.reset()
        done = False
        episode_steps = 0
        while not done:
            episode_steps += 1
            if total_steps<random_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action = (action + np.random.normal(0, noise, size=action.shape)).clip(-max_action, max_action)

            next_state, reward, done, _ = env.step(action)

            reward = reward + 20 * abs(next_state[1])

            if done and episode_steps != max_episode_steps:
                done = True
            else:
                done = False

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Take 50 steps,then update the networks 50 times
            if total_steps >= random_steps and total_steps % update_interval == 0:
                for _ in range(update_interval):
                    agent.learn(replay_buffer)
            
            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1)%evaluate_interval == 0:
                evaluate_num += 1
                evaluate_reward = evalueate_policy(env, agent, num_episodes=3)
                evaluate_rewards.append(evaluate_reward)

            total_steps += 1
            bar.update()

    bar.close()

#use imagif to record the video
def record_video(env, agent, out_directory, duration=20):
    with imageio.get_writer(out_directory, duration=duration) as video:
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            video.append_data(env.render(mode='rgb_array'))
            state = next_state

train_2()
record_video(env, agent, "ddpg_pendulum.gif", duration=20)
env.close()

