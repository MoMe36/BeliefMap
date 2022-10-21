from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v3
import ray.rllib.agents.sac as sac
import ray 

if __name__ == "__main__": 

    def env_creator(args):
        return PettingZooEnv(waterworld_v3.env())

    env = env_creator({})
    register_env("waterworld", env_creator)

    tuner = tune.run("APEX_DDPG", 
            stop= {'episode_total':1000}, 
            checkpoint_at_end = True, 
            local_dir = './testing_ray_marl', 
            config = {'env': 'waterworld',
                      'num_workers': 2, 
                      'multiagent': {'policies': set(env.env.agents), 
                                     'policy_mapping_fn': lambda agent_id, ep, **kwargs : agent_id}}) 