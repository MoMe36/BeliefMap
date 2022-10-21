import numpy as np 
import arcade 
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import time 

# =============== CHECK ===============
# https://colab.research.google.com/drive/1pRxOjSszukN5X0B7EO_YvGPEIZ7P-zdP#scrollTo=oFeuLIe26CZ4 
# =====================================


class TestFollow(MultiAgentEnv): 

    def __init__(self): 
        super().__init__()
        self.nb_agents = 3 
        self.radius = 0.3 
        self.leader_speed = 0.05 
        self.agent_speed = 1.1 * self.leader_speed
        self.leader_size = 0.05
        self.max_ts = 300
        self.action_space = gym.spaces.Box(-1.,1., shape = (2,))
        self.observation_space = gym.spaces.Box(low = -2., high = 2., shape = (6,))
        self._agent_ids = ['a{}'.format(i) for i in range(self.nb_agents)]

    def reset(self): 
        self.leader_angle = np.random.uniform(0., np.pi *2.)
        self.pos = np.random.uniform(0.1,0.9, size = (self.nb_agents, 2))
        self.ts = 0 
        self.leader_pos = self.compute_leader_pos()

        return self.get_obs()

    def compute_leader_pos(self): 
        return 0.5 + np.array([np.cos(self.leader_angle), np.sin(self.leader_angle)]) * self.radius
    def step(self, actions): 

        # print('=' * 5  + 'action'.upper() + '=' * 5)
        # input(actions)

        self.leader_angle = (self.leader_speed + self.leader_angle)%(np.pi*2)

        # self.pos = np.clip(self.pos + np.random.uniform(-1.,1., size =(self.nb_agents,2))* 0.02, 0.,1.)

        for i,k in enumerate(sorted(actions.keys())): 
            self.pos[i] = np.clip(self.pos[i] + self.agent_speed * actions[k], 0., 1.)

        self.ts += 1
        dones = {k : self.ts >= self.max_ts for k in self._agent_ids}
        dones['__all__'] = self.ts >= self.max_ts
        # print('Dones: {}'.format(dones))
        return self.get_obs(), self.compute_reward(), dones, {}
    def get_obs(self): 

        obs =  {}
        lp = self.compute_leader_pos()
        for i,k in enumerate(self._agent_ids):
            o = lp - self.pos[i]
            for j in range(self.nb_agents): 
                if j != i: 
                    o = np.hstack([o, self.pos[j] - self.pos[i]])
            obs[k] = o 

        # print('=' * 5  + 'obs'.upper() + '=' * 5)
        # for k in obs.keys() :
        #     print('{} : {}'.format(k, obs[k]))
        return obs

    def compute_reward(self): 
        r = {}
        lp = self.compute_leader_pos()
        for i,k in enumerate(self._agent_ids): 
            d = np.sqrt(np.sum((lp.flatten() - self.pos[i].flatten())**2))
            r[k] = np.exp(-3. * d)

        return r

    def draw(self): 
        for p in self.pos: 
            arcade.draw_circle_filled(*(p*600.), self.leader_size * 0.5 * 600, arcade.color.WISTERIA)
        arcade.draw_circle_outline(*(self.compute_leader_pos()*600.), self.leader_size * 600, arcade.color.RED, 3)


class CircleFollow(TestFollow): 
    def __init__(self): 
        super().__init__()
        self.offset_radius = 0.5

    def compute_reward(self): 
        lp = self.compute_leader_pos()
        v = lp - np.ones_like(lp) * 0.5
        target_pos = np.ones_like(lp) * 0.5 + v * (1. - self.offset_radius)
        r = {}
        for i,k in enumerate(self._agent_ids): 
            d = np.sqrt(np.sum(((self.pos[i].flatten() - target_pos.flatten())**2)))
            r[k]= np.exp(-d *3)
        return r


class ArcadeRender(arcade.Window): 

    def __init__(self, env): 
        super().__init__(600,600, "TestFollow")

        self.env = env
        self.background_color = arcade.color.WHITE
        self.world_state = self.env.reset()

    def compute_action(self, **kwargs): 
        return None

    def update(self, dt): 
        action = self.compute_action(obs = self.world_state)
        self.world_state, r, done, info = self.env.step(action)
        if done : 
            self.world_state = self.env.reset()
        return 

    def on_draw(self): 

        self.clear()
        self.env.draw()


class RayRender(ArcadeRender): 

    def set_agent(self, agent): 
        self.agent = agent 

    def update(self, dt): 

        action = {}
        for agent_id in self.world_state.keys(): 
            action[agent_id] = self.agent.compute_single_action(self.world_state[agent_id], policy_id = agent_id)         
        self.world_state, r, done, _ = self.env.step(action)

        print(r)
        time.sleep(0.1)
        if done['__all__']: 
            self.world_state = self.env.reset()




def train(env_func): 

    from ray.rllib.agents.sac import sac 
    from ray.rllib.agents.sac.sac_torch_policy import SACTorchPolicy
    from ray.rllib.agents.ppo import ppo 
    from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
    from ray import tune 
    import ray 


    def policy_mapping_fn(agent_id, episode, worker): 
        return agent_id

    def creator(env_config = None): 
        return env_func()

    ray.init()
    tune.register_env('follow', creator)

    config = ppo.DEFAULT_CONFIG.copy()
    config['train_batch_size'] = 4000
    config['framework'] = 'torch'
    config['num_workers'] = 4
    config['env'] = 'follow'

    policies = {}
    for k in env_func()._agent_ids: 
        policies[k] =  [PPOTorchPolicy, gym.spaces.Box(-2.,2., shape = (6,)), 
                                        gym.spaces.Box(-1.,1., shape = (2,)),
                                        {}]
    config['multiagent']['policies'] = policies
    config['multiagent']['policy_mapping_fn'] = policy_mapping_fn#lambda x : print('MAPPING ?? {}'.format(x))


    stop = {'timesteps_total' : 500000}
    resume = tune.run('PPO', 
        config = config, 
         stop = stop, 
         checkpoint_at_end = True, 
         local_dir = './ray_multi_xps')

    print(resume.get_last_checkpoint())
    ray.shutdown()



def test(env_func): 

    from ray.rllib.agents.sac import sac 
    from ray.rllib.agents.sac.sac_torch_policy import SACTorchPolicy
    from ray.rllib.agents.ppo import ppo 
    from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
    from ray import tune 
    import ray 


    def policy_mapping_fn(agent_id, episode, worker): 
        return agent_id

    def creator(env_config = None): 
        return env_func()

    ray.init()
    tune.register_env('follow', creator)

    config = ppo.DEFAULT_CONFIG.copy()
    config['env'] = 'follow'
    config['framework'] = 'torch'
    policies = {}
    for k in env_func()._agent_ids: 
        policies[k] =  [PPOTorchPolicy, gym.spaces.Box(-2.,2., shape = (6,)), 
                                        gym.spaces.Box(-1.,1., shape = (2,)),
                                        {}]
    config['multiagent']['policies'] = policies
    config['multiagent']['policy_mapping_fn'] = policy_mapping_fn#lambda x : print('MAPPING ?? {}'.format(x))
    agent = ppo.PPOTrainer(config = config)
    agent.restore('/home/mehdi/Codes/PostPhD/BeliefMap/ray_multi_xps/PPO/PPO_follow_d4118_00000_0_2022-10-17_08-56-30/checkpoint_000125/checkpoint-125')
    

    render= RayRender(env_func())
    render.set_agent(agent)
    arcade.run()


if __name__ == "__main__": 

    # ArcadeRender(TestFollow())
    # arcade.run()

    train(CircleFollow)
    test(TestFollow)