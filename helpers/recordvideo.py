import time
from typing import Dict, Any

import gym
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.logger import Image
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env or VecEnv, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True, modelname = "ppo_cnn"):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super(VideoRecorderCallback, self).__init__(verbose=2)
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self.model_name = modelname
        self._cnt_stps = 0

    def _on_training_start(self) -> None:
        super()._on_training_start()
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        super()._on_step()
        image = self.training_env.render(mode="rgb_array")
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout",  "csv"))
        if self._cnt_stps % self._render_freq == 0:
            # screens = []

            # def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
            #     """
            #     Renders the environment in its current state, recording the screen in the captured `screens` list

            #     :param _locals: A dictionary containing all local variables of the callback's scope
            #     :param _globals: A dictionary containing all global variables of the callback's scope
            #     """
            #     screen = self._eval_env.render(mode="rgb_array")
            #     # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
            #     screens.append(screen.transpose(2, 0, 1))

            # evaluate_policy(
            #     self.model,
            #     self._eval_env,
            #     callback=grab_screens,
            #     n_eval_episodes=self._n_eval_episodes,
            #     deterministic=self._deterministic,
            # )
            # clip = Video(torch.ByteTensor([screens]), fps=40)
            # np.save(f"./videos/video_{self.num_timesteps}_{self.model_name}.npy", clip)
            print(f"Saving model at {self.num_timesteps}")
            self.model.save(f"./models/model_{self.num_timesteps}_{self.model_name}.zip")

            # self.logger.record(
            #     "trajectory/video",
            #     clip,
            # )
            

        self._cnt_stps += 1
        self.logger.dump(self.num_timesteps)
        return True
        
    def _on_rollout_end(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),

        )
    

