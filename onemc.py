from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from helpers.minedojo_hunt_cow_gymnasium_wrapper import get_openai_env, multiproc
import sys
import gym
from stable_baselines3 import PPO
import minedojo
from helpers.recordvideo import VideoRecorderCallback

env = minedojo.make(
    task_id="harvest_wool_with_shears_and_sheep",
    image_size=(160, 256)
)
obs = env.reset()
for i in range(2):
    act = env.action_space.no_op()
    act[0] = 1    # forward/backward
    if i % 10 == 0:
        act[2] = 1    # jump
    obs, reward, done, info = env.step(act)
env.close()

num_cpu = 6
env = SubprocVecEnv([multiproc(i) for i in range(num_cpu)], start_method='fork')
# env = VecEnvWrapper(env)
env = VecFrameStack(env, n_stack=8)  # frame stacking for temporal information
eval_env =SubprocVecEnv([multiproc(num_cpu + 5)], start_method='fork')
eval_env = VecFrameStack(eval_env, 8)


# env = multiproc(0)()
# Our action space:
# 0: move forward/back (0: noop, 1: forward, 2: backward)
# 1: turn left/right (0: noop, 1: left, 2: right)
# 2: jump/sprint/crouch (0: noop, 1: jump, 2: sneak, 3:sprint)
# 3: camera pitch (0: -180 degree, 24: 180 degree) (25 values)
# 4: camera yaw (0: -180 degree, 24: 180 degree) (25 values)
# 5: attack (0: noop, 1: attack)

# Train the model using PPO2
log_freq = 300
modelname = "PPO_CNN"
game = "MINECRAFT_HUNT_COW_DENSE"
log_path = "./logs"

callback = CallbackList([ EvalCallback(eval_env, best_model_save_path=f"{log_path}{modelname}{game}best_model",
                                log_path=f"{log_path}{modelname}results", eval_freq=log_freq, render=False)])

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./logs/mcPPO/")

model.learn(total_timesteps=100000, log_interval=10, tb_log_name="minecraft")
model.save("ppo2_multidiscrete")
