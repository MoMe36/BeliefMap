import gym 
# import sys 
# sys.path.append('./')
# import belief_envs    
from stable_baselines3.common.env_checker import check_env 
# gym.make('belief-v0')
from env_belief import BeliefEnv
env = BeliefEnv()

from stable_baselines3 import PPO
m = PPO('MultiInputPolicy', env, verbose= 1)

m.learn(total_timesteps = 3000) 