from torch import nn
import torch
import gym
import os
from collections import deque
import itertools
import numpy as np
import random
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack,PytorchLazyFrames
import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

gamma = 0.99

def model(observation_space):
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, 32, kernel_size = 4, stride =4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size =4, stride =2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size =3, stride =1),
        nn.ReLU(),
        nn.Flatten())

    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(cnn, nn.Linear(n_flatten, 512), nn.ReLU())

    return out

class Network(nn.Module):
    def __init__(self, env, device):
        super().__init__()

        self.num_actions = env.action_space.n
        self.device = device
        conv_net = model(env.observation_space)
        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))


    def forward(self, x):
        return self.net(x)

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device = self.device)
        q_values = self(obses_t)
        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <=epsilon:
                actions[i] = random.randint(0, self.num_actions-1)
        return actions

    def compute_loss(self, transitions, target_net):
        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[4] for t in transitions]

        if isinstance(obses[0], PytorchLazyFrames):
            obses = np.stack([o.get_frames() for o in obses])
            new_obses = np.stack([o.get_frames() for o in new_obses])

        else:
            obses = np.asarray(obses)
            new_obses = np.asarray(new_obses)

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device = self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device = self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device = self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device = self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device = self.device)

        # Compute Targets
        # targets = r + gamma * target q vals * (1 - dones)
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + gamma * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss, action_q_values

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k,t in self.state_dict().items()}
        params_data = msgpack.dumps(params)

        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path,'rb') as f:
            params_numpy = msgpack.loads(f.read())

        params = {k: torch.as_tensor(v, device = self.device) for k,v in params_numpy.items()}

        self.load_state_dict(params)

       