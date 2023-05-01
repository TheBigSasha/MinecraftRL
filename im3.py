import multiprocessing
import types

import gym
import numpy as np
from imitation.algorithms import bc
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.bc import reconstruct_policy
import minerl
from minerl.data import download
from minerl.data import BufferedBatchIter
from sklearn.cluster import KMeans
import tqdm

# protect the entry point
if __name__ == '__main__':
    # enable support for multiprocessing
    multiprocessing.freeze_support()

    NUM_CLUSTERS = 32

    download(directory='./minerldata', environment="MineRLTreechopVectorObf-v0")

    # Load the dataset
    data = minerl.data.make("MineRLTreechopVectorObf-v0", data_dir="./minerldata")

    # Load the dataset storing 1000 batches of actions
    act_vectors = []
    iterator = BufferedBatchIter(data)


    for obs, act, _, _, _ in tqdm.tqdm(data.batch_iter(16, 32, 2, preload_buffer_size=20)):
        act_vectors.append(act['vector'])
        if len(act_vectors) > 1000:
            break

    obs_shape = (64, 64, 3)
    # Reshape these the action batches
    acts = np.concatenate(act_vectors).reshape(-1, 64)
    kmeans_acts = acts[:100000]

    # Use sklearn to cluster the demonstrated actions
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(kmeans_acts)

    # Save the kmeans model
    import pickle
    with open('kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)


    # Custom environment with KMeans clustered action space
    class KMeansTreechopEnv(gym.Env):
        def __init__(self):
            self.env = gym.make('MineRLTreechopVectorObf-v0')
            self.action_space = gym.spaces.Discrete(NUM_CLUSTERS)
            self.observation_space = gym.spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8)

        def step(self, action):
            kmeans_action = {'vector': kmeans.cluster_centers_[action]}
            obs, reward, done, info = self.env.step(kmeans_action)
            return obs['pov'], reward, done, info

        def reset(self):
            return self.env.reset()

        def render(self, mode='human'):
            return self.env.render(mode)


    # Collect expert demonstrations
    kmeans_env = KMeansTreechopEnv()
    demonstration_data = []

    for obs, act, rew, next_obs, done in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):
        # print("obs: ", obs)
        action_cluster = kmeans.predict(act['vector'].reshape(1, -1))
        transition = {'obs': obs['pov'], 'acts': action_cluster, 'rew': rew, 'next_obs': next_obs['pov'], 'done': done}
        demonstration_data.append(transition)

    print(demonstration_data[0]['obs'].shape)

    demonstrations = demonstration_data

    # print("Number of demonstrations: ", len(demonstrations))
    # print("Example demonstration: ", demonstrations[0])
    # print("Example demonstration: ", demonstrations[1])

    # Prepare expert demonstrations
    rng = np.random.default_rng(0)

    # Train BC model
    bc_trainer = bc.BC(
        batch_size=1,
        observation_space=kmeans_env.observation_space,
        action_space=kmeans_env.action_space,
        demonstrations=demonstrations,
        rng=rng
    )

    # Train the model
    bc_trainer.train(n_epochs=2)

    bc_trainer.save_policy("im3_policy.pkl")

    # load the policy
    policy = reconstruct_policy("im3_policy.pkl")

    # Evaluate the trained policy
    wrapped_env = DummyVecEnv([lambda: kmeans_env])
    model = PPO("CnnPolicy", wrapped_env, verbose=1, tensorboard_log="./tensorboard/")
    model.policy = bc_trainer.policy
    #
    model.save("im3_model.zip")
    # model = PPO.load("im3_model.zip")

    mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=10)
    print("Mean reward:", mean_reward, "Std reward:", std_reward)
