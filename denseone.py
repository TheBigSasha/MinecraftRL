import sys
from logging import getLogger

import gym
import torch

logger = getLogger(__name__)

sys.path.insert(0, '..')

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO


class Args:
    def __init__(self):
        self.frame_skip = None
        self.gray_scale = False
        self.env = 'MineRLNavigateDense'
        self.frame_stack = None
        self.disable_action_prior = False


args = Args()

core_env = gym.make("MineRLNavigateDense-v0")
env = core_env


class ModifiedCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super(ModifiedCNN, self).__init__(observation_space, features_dim)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(20736, 512),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)


class CustomPolicy(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomPolicy, self).__init__()

        self.features_extractor = ModifiedCNN(observation_space)

    def forward(self, observations):
        return self.features_extractor(observations)


policy_kwargs = dict(
    features_extractor_class=ModifiedCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./test_tensorboard/")

model.learn(total_timesteps=300000, log_interval=100)
model.save("level1.5_ppo3")
