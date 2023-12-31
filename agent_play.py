import numpy as np
import torch
import itertools
from baselines_wrappers import DummyVecEnv, SubprocVecEnv
from network import Network
from dqn_update import SAVE_PATH
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
import time
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



make_env= lambda: make_atari_deepmind('BreakoutNoFrameskip-v4')
vec_env = DummyVecEnv([make_env for _ in range(1)])
env=BatchedPytorchFrameStack(vec_env, k=4)

net=Network(env, device)
net=net.to(device)
net.load(SAVE_PATH)

obs=env.reset()
beginning_episode=True

for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = net.act(act_obs, 0.0)
    else:
        action = net(obs, 0.0)

    if beginning_episode:
        action=[1]
        beginning_epsidoe=False

    obs, rew, done, _ = env.step(action)
    env.render()
    time.sleep(0.2)

    if done[0]:
        obs = env.reset()
        beginning_epsidoe=True

