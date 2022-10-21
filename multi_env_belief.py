import numpy as np 
import gym 
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer 

class TestEnv(MultiAgentEnv): 
    def __init__(self, nb_agents= 5): 
        super().__init__()
        print(nb_agents)
        self.nb_agents = nb_agents
        self.map = np.zeros((36,36))

    def reset(self): 
        self.pos = np.zeros((self.nb_agents, 2)) + np.arange(self.nb_agents).reshape(-1,1) * 0.1
        self.ts = 0

        return self.get_obs()

    def step(self, action_dict): 

        self.ts += 1
        return self.get_obs(), self.compute_rewards(), self.get_dones(), {}

    def get_dones(self): 
        done = True if self.ts > 200 else False
        return {'a{}'.format(i): done for i in range(self.nb_agents)}

    def compute_rewards(self): 
        rewards = {}
        for i in range(self.pos.shape[0]): 
            rewards['a{}'.format(i)] = i * 0.1
        return rewards

    def get_obs(self): 
        obs = {}
        for i in range(self.pos.shape[0]): 
            obs['a{}'.format(i)] = (self.map, self.pos[i].flatten())

        return obs

if __name__ == "__main__": 

    env = TestEnv()
    trainer = PPOTrainer(env = TestEnv, config = {'env_config' : {"nb_agents":2 }})