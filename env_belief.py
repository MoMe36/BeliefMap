import numpy as np 
import arcade 
import gym 
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
from tqdm import tqdm 
plt.style.use('ggplot')
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

class BeliefEnv(gym.Env): 

    def __init__(self):

        
        self.nb_cells = 36
        self.world_size = 600
        self.cell_size = self.world_size/self.nb_cells
        self.forget_speed = 1.5e-3
        self.vision_radius = 0.15
        self.max_speed = 1e-2
        self.belief_map = np.zeros([self.nb_cells]*2)

        self.action_space = gym.spaces.Box(low = -1.,high = 1., shape = (2,))
        obs_spaces = {'img': gym.spaces.Box(low = 0, high = 255, 
                                            shape = (self.nb_cells, self.nb_cells, 1), 
                                            dtype = np.uint8), 
                     'vec': gym.spaces.Box(low = -1., high = 1., shape = (2,))}
        self.observation_space =gym.spaces.Dict(obs_spaces)# obs_spaces#['vec'] #


    def draw_beliefs(self): 

        known_color = np.array([1.,1.,1.]) * 255
        unknown_color = np.array((0,150,50))

        for i in range(self.belief_map.shape[0]): 
            for j in range(self.belief_map.shape[1]):
                p = np.array([i,j], dtype = float) * self.cell_size
                p += np.ones_like(p) * 0.5 * self.cell_size

                alpha = np.clip(self.belief_map[i,j],0.,1.)
                belief_color = known_color * alpha + (1. - alpha) * unknown_color
                arcade.draw_rectangle_filled(*p, self.cell_size * 0.93, self.cell_size * 0.93,
                                             belief_color)           
               
        arcade.draw_text("Coverage: {:.2f}\nReward: {:.4f}\nTS: {}/{}".format(self.coverage, self.compute_reward(), self.ts, self.max_ts), 20, 30, arcade.color.BLACK, 20)

    def draw(self): 
        self.draw_beliefs()
        arcade.draw_rectangle_filled(*(self.pos*600), 20,20, (250,0,0))

    def compute_reward(self): 

        r = self.coverage - self.prev_coverage
        if np.sum(np.not_equal(np.clip(self.pos,0.,1.), self.pos))> 0: 
            return -5.
        else: 
            if r == 0: 
                return -0.1
            else: 
                return r * 100. + 0.3 * np.exp(-3. * (1. - self.coverage))

    def compute_coverage(self): 
        coverage = np.sum(np.where(self.belief_map > 0.2, 1. ,0.).flatten()) / np.prod(self.belief_map.shape)
        return coverage

    def get_obs(self): 

        self.prev_coverage = self.coverage
        self.coverage = self.compute_coverage()

        # return (self.belief_map.reshape(*self.belief_map.shape, 1) * 255).astype(np.uint8)
        # return np.array([0,0])
        return {'img': (self.belief_map.reshape(*self.belief_map.shape, 1) * 255).astype(np.uint8), 
                'vec': self.pos}
    def reset(self): 

        self.pos = np.random.uniform(0.1,0.9, size = (2,))#np.random.uniform(0.,1.,size = (2,)) * 0. +  np.array([0.1 if np.random.uniform() < 0.5 else 0.9, 0.1 if np.random.uniform() < 0.5 else 0.9])
        self.ts = 0
        self.max_ts = 500
        self.belief_map *= 0. 

        self.update_beliefs()
        self.coverage = self.compute_coverage()
        self.prev_coverage = 0. 

        obs = self.get_obs()
        return obs

    def update_beliefs(self): 

        pos_vec = np.arange(self.nb_cells).reshape(-1,1)
        
        x = np.repeat(pos_vec, self.nb_cells, axis = 0)
        y = np.tile(pos_vec,(self.nb_cells,1)).reshape(-1,1)
        pos_map = np.hstack([x,y]) * self.cell_size + 0.5 * self.cell_size
    
        dists = np.sqrt(np.sum((self.pos * 600 - pos_map)**2, 1))
        idx = np.where(dists < self.vision_radius * 600)[0]

        visible = np.hstack([x,y])[idx]

        self.belief_map = np.clip(self.belief_map - self.forget_speed, 0.,1.)
        self.belief_map[visible[:,0], visible[:,1]] = 1. 


    def step(self, offset): 
        self.ts += 1
        self.pos += offset * self.max_speed
        # self.pos = np.clip(self.pos, 0.,1.)

        self.update_beliefs()

        obs = self.get_obs()
        r = self.compute_reward()
        done = self.is_done()

        return obs, r, done, {'coverage': self.compute_coverage()}

    def is_done(self): 
        return self.ts >= self.max_ts or np.sum(np.not_equal(np.clip(self.pos,0.,1.), self.pos))> 0

    def render(self): 
        return

class BeliefRLLib(BeliefEnv): 
    def get_obs(self): 
        obs = super().get_obs()
        img = obs['img'].astype(float) / 255.
        return {'img': img, 'vec': obs['vec']}


class RandomModel: 
    def predict(self, obs): 
        return np.random.uniform(-1.,1.,  size = (2,)), None

def test_agent(env, model, nb_eps = 10): 

    # env = BeliefEnv()
    coverage_rate = []
    rewards = []
    pbar = tqdm(total = nb_eps)
    for i in range(nb_eps): 
        ep_coverage = []
        reward = 0.
        s = env.reset()
        done = False 
        while not done: 
            if hasattr(model, "predict"):
                action = model.predict(s)[0]
            else:
                action = model.compute_action(s)

            s, r , done, info = env.step(action)
            ep_coverage.append(info['coverage'])
            reward += r 

        coverage_rate.append(ep_coverage)
        rewards.append(reward)
        pbar.update(1)
    pbar.close()

    f, axes = plt.subplots(1,2)
    axes = axes.flatten()

    for cr in coverage_rate: 
        axes[0].plot(cr, color = 'red',  alpha = 0.3)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Coverage')
    axes[0].set_ylim(0.,1.)
    axes[0].set_title('Coverage over episode', weight = 'bold')
    axes[1].hist(rewards)
    axes[1].set_title("Cumulative reward distribution", weight = 'bold')

    plt.savefig('./latest_{}.png'.format(type(env)))
    # plt.show()



class ArcadeRender(arcade.Window): 

    def __init__(self, belief_env): 
        super().__init__(600,600, "Belief")

        self.belief_env = belief_env
        self.background_color = arcade.color.BLACK
        self.world_state = self.belief_env.reset()

    def compute_action(self, **kwargs): 
        return np.random.uniform(-1.,1.,size = (2,))

    def update(self, dt): 
        action = self.compute_action(obs = self.world_state)
        self.world_state, r, done, info = self.belief_env.step(action)
        if self.belief_env.is_done(): 
            self.world_state = self.belief_env.reset()
        return 

    def on_draw(self): 

        self.clear()
        self.belief_env.draw()

class AgentRender(ArcadeRender): 
    def __init__(self, agent): 
        super().__init__(BeliefEnv())
        self.agent = agent
    def compute_action(self, obs): 
        action = self.agent.predict(obs)[0]


        return action


class RLLibRender(ArcadeRender): 
    def __init__(self, agent): 
        super().__init__(BeliefRLLib())
        self.agent = agent 

    def compute_action(self, obs): 
        action = self.agent.compute_action(obs)

        return action


# class CustomTorch(TorchModelV2): 
#     def __init__(self): 


def ray_train(): 
    from ray.rllib.agents.ppo import PPOTrainer
    import ray.rllib.agents.ppo as ppo 
    from ray.tune.registry import register_env
    import ray 
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray import tune 
    from ray.tune.logger import pretty_print

    # ModelCatalog.register_custom_model("custom", CustomTorch)

    ray.init()

    def env_creator(env_config): 
        return BeliefRLLib()


    register_env("belief", env_creator)
    config = ppo.DEFAULT_CONFIG.copy()
    config['model'] = {'conv_filters': [[32, [8,8], 4], 
                                    [64, [4,4], 2], 
                                    [64, [3,3], 1]], 
                        
                        }
    config['framework'] = 'torch'
    config['env'] = 'belief'
    config['num_workers'] = 4
    config['num_envs_per_worker'] = 3
    # config['train_batch_size'] = 128
    config['ignore_worker_failures']=True
    config['recreate_failed_workers']=True 

    # agent = PPOTrainer(config = config, env = "belief")
    # for i in range(100): 
    #     result = agent.train()
    #     pretty_print(result)
    #     if result['timesteps_total'] > 5000: 
    #         break 
    # path = agent.save()
    # print('Checkpoint: {}'.format(path))

    stop = {'timesteps_total':1000000 }
    results = tune.run("PPO", 
                config = config, 
                stop = stop, 
                checkpoint_at_end = True, 
                local_dir = './ray_xps')

    ray.shutdown()


def ray_test():
    from ray.rllib.agents.ppo import PPOTrainer
    import ray.rllib.agents.ppo as ppo 
    from ray.tune.registry import register_env
    import ray 
    from ray import tune 


    ray.init()

    def env_creator(env_config): 
        return BeliefRLLib()


    register_env("belief", env_creator)
    config = ppo.DEFAULT_CONFIG.copy()
    config['model'] = {'conv_filters': [[32, [8,8], 4], 
                                    [64, [4,4], 2], 
                                    [64, [3,3], 1]],}
    config['framework'] = 'torch'
    config['env'] = 'belief'
    # config['train_batch_size'] = 128
    config['ignore_worker_failures']=True
    config['recreate_failed_workers']=True 

    agent = PPOTrainer(config= config)
    path = "/home/mehdi/Codes/PostPhD/BeliefMap/ray_xps/PPO/PPO_belief_456a9_00000_0_2022-07-21_20-16-42/checkpoint_000126/checkpoint-126"

    agent.restore(path)
    
    test_agent(BeliefRLLib(), agent)

    RLLibRender(agent)
    arcade.run()


    ray.shutdown()
    # agent.restore('./ray_xps/PPO/PPO_belief_d2417_00000_0_2022-07-20_09-16-06/checkpoint_000001/checkpoint-1')

if __name__ == "__main__": 

    # ray_train()
    ray_test()

    # ArcadeRender(BeliefEnv())
    # arcade.run()
    # test_agent()


    # AgentRender()
    # arcade.run()


    # envs = SubprocVecEnv([BeliefEnv for i in range(8)])
    # m = PPO('MultiInputPolicy', VecMonitor(venv = envs), verbose = 1, device = 'cpu')
    # m.learn(total_timesteps = 1000000)
    # m.save('test_ppo')




    # m = PPO.load('./test_ppo')
    # test_agent(m)
    # AgentRender(m)
    # arcade.run()






