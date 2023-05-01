import multiprocessing

import gym
import numpy as np
from gym.spaces import MultiDiscrete
from imitation.algorithms import bc
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.bc import reconstruct_policy
import minerl
from minerl.data import download
from minerl.data import BufferedBatchIter
import tqdm

# protect the entry point
if __name__ == '__main__':
    # enable support for multiprocessing
    multiprocessing.freeze_support()

    download(directory='./minerldata', environment="MineRLTreechopVectorObf-v0")

    # Load the dataset
    data = minerl.data.make("MineRLTreechopVectorObf-v0", data_dir="./minerldata")

    # Load the dataset storing 1000 batches of actions
    act_vectors = []
    iterator = BufferedBatchIter(data)

    for obs, act, _, _, _ in tqdm.tqdm(data.batch_iter(16, 32, 2, preload_buffer_size=20)):
        act_vectors.append(act)
        if len(act_vectors) > 1000:
            break

    obs_shape = (64, 64, 3)

    # Custom environment without KMeans clustered action space
    class TreechopEnv(gym.Env):
        def __init__(self):
            self.env = gym.make('MineRLTreechopVectorObf-v0')
            self.action_space = self._convert_to_multidiscrete(self.env.action_space)
            self.observation_space = gym.spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8)

        def _convert_to_multidiscrete(self, action_space):
            action_limits = []
            for key, space in action_space.spaces.items():
                if isinstance(space, gym.spaces.Discrete):
                    action_limits.append((0, space.n - 1))
                elif isinstance(space, gym.spaces.Box):
                    lower_limit = max(0, int(space.low[0]))
                    action_limits.append((lower_limit, int(space.high[0])))

            # Print action_limits for debugging
            print("Action limits:", action_limits)

            return MultiDiscrete(action_limits)

        def _convert_multidiscrete_to_dict(self, action):
            action_dict = {}
            for i, key in enumerate(self.env.action_space.spaces.keys()):
                action_dict[key] = action[i]
            return action_dict

        def step(self, action):
            action_dict = self._convert_multidiscrete_to_dict(action)
            obs, reward, done, info = self.env.step(action_dict)
            return obs['pov'], reward, done, info

        def reset(self):
            return self.env.reset()

        def render(self, mode='human'):
            return self.env.render(mode)

    # Collect expert demonstrations
    env = TreechopEnv()
    demonstration_data = []

    for obs, act, rew, next_obs, done in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):
        transition = {'obs': obs['pov'], 'acts': act, 'rew': rew, 'next_obs': next_obs['pov'], 'done': done}
        demonstration_data.append(transition)

    print(demonstration_data[0]['obs'].shape)

    demonstrations = demonstration_data

    # Prepare expert demonstrations
    rng = np.random.default_rng(0)

    # Train BC model
    bc_trainer = bc.BC(
        batch_size=1,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=demonstrations,
        rng=rng
    )

    # Train the model
    bc_trainer.train(n_epochs=2)

    bc_trainer.save_policy("im3_policy.pkl")

    # load the policy
    policy = reconstruct_policy("im3_policy.pkl")

    # Evaluate the trained policy
    wrapped_env = DummyVecEnv([lambda: env])
    model = PPO("CnnPolicy", wrapped_env, verbose=1, tensorboard_log="./tensorboard/")
    model.policy = bc_trainer.policy

    model.save("im3_model.zip")
    # model = PPO.load("im3_model.zip")

    mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=10)
    print("Mean reward:", mean_reward, "Std reward:", std_reward)
