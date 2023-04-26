from typing import Callable

import gym
import numpy as np
import time

from helpers.hunt_cow import HuntCowDenseRewardEnv


def get_openai_env(step_penalty: float or int,
                   nav_reward_scale: float or int,
                   attack_reward: float or int,
                   success_reward: float or int,
                   image_size: tuple[int, int],
                   suffix: str = ''):
    class MyEnv(gym.Env):
        def __init__(self):
            self._env = HuntCowDenseRewardEnv(
                step_penalty=step_penalty,
                nav_reward_scale=nav_reward_scale,
                attack_reward=attack_reward,
                success_reward=success_reward,
                image_size=image_size,
            )
            self._saved_images = []
            self._cnt_im = 0

        def _save_im(self, x):
            self._saved_images.append(x)
            if len(self._saved_images) > 400:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                np.save(f'./saved/saved_images_{self._cnt_im}_{timestr}_{suffix}.npy', np.array(self._saved_images))
                self._cnt_im += 1
                self._saved_images = []

        def reset(self, seed=None, **kwargs):
            if seed is not None:
                self._env.seed(seed)
            res = self._env.reset(**kwargs, onadd=self._save_im)
            return res

        def step(self, action):
            return self._env.step(action, onadd=self._save_im)

        def render(self, mode='human', **kwargs):
            return self._env.render(mode, **kwargs)

        def close(self):
            self._env.close()

        def seed(self, seed=None):
            self._env.seed(seed)

        @property
        def action_space(self):
            return self._env.action_space

        @property
        def observation_space(self):
            return self._env.observation_space

    return MyEnv


def multiproc(i) -> Callable:
    def _init() -> gym.Env:
        env = get_openai_env(
            step_penalty=0.1,
            nav_reward_scale=2,
            attack_reward=10,
            success_reward=100,
            image_size=(128, 180),
            suffix=f'_{i}',
        )()
        env.reset(seed=i * 1000)
        return env

    return _init
