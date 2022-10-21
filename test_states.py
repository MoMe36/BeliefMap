from stable_baselines3 import PPO 
from stable_baselines3.common.envs import SimpleMultiObsEnv


env = SimpleMultiObsEnv(random_start = False)
s = env.reset()

for k in s.keys(): 
    print(s[k].shape) 