from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
import gym
import os
from collections import deque
import itertools
import numpy as np
import random
from network import Network
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack,PytorchLazyFrames
from baselines_wrappers import DummyVecEnv, Monitor
import msgpack


gamma = 0.99
batch_size = 32
BUFFER_SIZE=int(1e6)
MIN_REPLAY_SIZE=50000
ep_start=1.0
ep_end=0.01 
NUM_ENVS = 4
ep_decay = int(1e6)
TARGET_UPDATE_FREQ = 2500 
LR = 5e-5
SAVE_PATH = './atari_model.pack'
SAVE_INTERVAL = 10000
LOG_INTERVAL = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
#  BreakoutNoFrameskip-v4, Seaquest-v4, SpaceInvaders-v4
make_env = lambda: Monitor(make_atari_deepmind('SpaceInvaders-v4', scale_values = True), allow_early_resets=True)
vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
env = BatchedPytorchFrameStack(vec_env, k=4)

replay_buffer = deque(maxlen=BUFFER_SIZE)
epinfos_buffer = deque([], maxlen=100)

episode_count = 0


online_net = Network(env, device = device)
target_net = Network(env, device = device)

online_net = online_net.to(device)
target_net = target_net.to(device)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

# Initialize replay buffer
obses = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    actions = [env.action_space.sample() for _ in range(NUM_ENVS)]

    new_obses, rews, dones, infos = env.step(actions)

    for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)

    obses = new_obses


# Main Training Loop
reward_store = []
episode_reward = []
ACTION_Q_VAL = []
obses = env.reset()
for step in itertools.count():
    epsilon = np.interp(step*NUM_ENVS, [0, ep_decay], [ep_start, ep_end])

    rnd_sample = random.random()

    if isinstance(obses[0], PytorchLazyFrames):
        act_obses = np.stack([o.get_frames() for o in obses])
        actions = online_net.act(act_obses, epsilon)
    else:
        actions = online_net(obses, epsilon)

    new_obses, rews, dones, infos = env.step(actions)
    if reward_store == []:
        reward_store = np.array([rews])
    else:
        reward_store = np.vstack([reward_store,rews])

    for obs, action, rew, done, new_obs, info in zip(obses, actions, rews, dones, new_obses, infos):
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)

        if done:
            epinfos_buffer.append(info['episode'])
            episode_count += 1
            if episode_reward == []:
                episode_reward = np.sum(reward_store)
            else:
                episode_reward = np.vstack([episode_reward,np.sum(reward_store)])
            np.savez('episode_reward.npz', epi_rew = episode_reward)
            reward_store = []
            

    obses = new_obses

    # start gradient descent
    transitions = random.sample(replay_buffer, batch_size)
    loss, action_q_val = online_net.compute_loss(transitions, target_net)

    action_q = action_q_val.cpu().detach().numpy()

    if ACTION_Q_VAL == []:
        ACTION_Q_VAL = np.array(action_q)
    else:
        ACTION_Q_VAL = np.vstack([ACTION_Q_VAL,np.array(action_q)])

    np.savez('q_val.npz', q_val = ACTION_Q_VAL)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # Update Target Net
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % LOG_INTERVAL == 0:
        rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
        len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0
        print()
        print('Step:', step)
        print('Avg Rew:', rew_mean)
        print('Avg Ep len', len_mean)
        print('Episodes',episode_count)
    # save
    if step%SAVE_INTERVAL == 0 and step != 0:
        print('saving...')
        online_net.save(SAVE_PATH)










