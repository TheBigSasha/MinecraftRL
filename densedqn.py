import gym

from logging import getLogger

from stable_baselines3.dqn.policies import DQNPolicy

logger = getLogger(__name__)

import sys
import minerl

sys.path.insert(0, '..')

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

class Args:
    def __init__(self):
        self.frame_skip = None
        self.gray_scale = False
        self.env = 'MineRLNavigateDense'
        self.frame_stack = None
        self.disable_action_prior = False


args = Args()


def wrap_env(env, test):
    # [Your existing wrap_env code without changes]
    return env


core_env = gym.make("MineRLNavigateDense-v0")
env = wrap_env(core_env, test=False)

# Ensure the environment is a vectorized environment
env = DummyVecEnv([lambda: env])

# Create the model
model = DQN('CnnPolicy', env, verbose=1, tensorboard_log="./test_tensorboard/")

# Train the model
model.learn(total_timesteps=100000, log_interval=100)
model.save("level1pt5_dqn")
