import os
import pybullet as p
import pybullet_data
from safe_rl import ppo_lagrangian, ppo
import gym
import gym_panda
from datetime import datetime


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M_")
file_name = "ppo_action1-EpLen500-spe1000-tip.robot"

ppo(
	env_fn = lambda : gym.make('panda-v0'),
	ac_kwargs = dict(hidden_sizes=[64,64]),
	logger_kwargs = dict(output_dir='exp-results/'+current_time+file_name, exp_name=(file_name)),
	steps_per_epoch=1000, # default: 4000
    epochs=200,
    max_ep_len=500,
	)


