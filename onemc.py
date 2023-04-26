from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from helpers.minedojo_hunt_cow_gymnasium_wrapper import get_openai_env, multiproc
import sys
import gym
from stable_baselines3 import PPO

# Create the custom environment
# Cls_env = get_openai_env(
#     step_penalty=0.1,
#     nav_reward_scale=2,
#     attack_reward=10,
#     success_reward=100,
#     image_size=(128, 180),
# )
#
# env = Cls_env()
num_cpu = 1
env = multiproc(0)
# env = SubprocVecEnv([multiproc(i) for i in range(num_cpu)], start_method='fork')
# env = VecFrameStack(env, n_stack=8)  # frame stacking for temporal information

# Our action space:
# 0: move forward/back (0: noop, 1: forward, 2: backward)
# 1: turn left/right (0: noop, 1: left, 2: right)
# 2: jump/sprint/crouch (0: noop, 1: jump, 2: sneak, 3:sprint)
# 3: camera pitch (0: -180 degree, 24: 180 degree) (25 values)
# 4: camera yaw (0: -180 degree, 24: 180 degree) (25 values)
# 5: attack (0: noop, 1: attack)

# Train the model using PPO2
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo2_minedojo_tensorboard/")

model.learn(total_timesteps=9000, log_interval=10, tb_log_name="first_run")
model.save("ppo2_multidiscrete")

# Load the model and enjoy the trained agent
del model
model = PPO.load("ppo2_multidiscrete")

model.set_env(env)
model.learn(total_timesteps=25000, log_interval=10, tb_log_name="second_run")
model.save("ppo2_multidiscrete_2")

del model
model = PPO.load("ppo2_multidiscrete_2")
model.set_env(env)
model.learn(total_timesteps=100000, log_interval=10, tb_log_name="third_run")
model.save("ppo2_multidiscrete_3")

del model
model = PPO.load("ppo2_multidiscrete_3")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
