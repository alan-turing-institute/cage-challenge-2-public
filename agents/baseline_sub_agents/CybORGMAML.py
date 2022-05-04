import os.path as path
from gym import spaces
import gym
#from agents.baseline_sub_agents.scaffold_env import *
import ray.rllib.agents.ppo as ppo
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.env.env_context import EnvContext
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import inspect
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG import CybORG
import os
from CybORGAgent import CybORGAgent
import random
from sub_agents import sub_agents
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG

class TorchModel(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                 name)
        torch.nn.Module.__init__(self)

        self.model = TorchFC(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class CybORGMAMLWrapped(gym.Env):
    # Env parameters
    max_steps = 100 # Careful! There are two other envs!
    mem_len = 1

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):

        self.cyborg = CybORG(self.path, 'sim', agents={'Red':RedMeanderAgent})
        self.RMenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.cyborg = CybORG(self.path, 'sim', agents={'Red':B_lineAgent})
        self.BLenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.adversary = None
        self.set_task(self.sample_tasks(1)[0])
        #relative_path = #'cage-challenge-1' #[:62], os.path.abspath(os.getcwd()) +
        #print(relative_path)



        self.steps = 0
        self.agent_name = 'MAML'

        #action space is 2 for each trained agent to select from
        self.action_space = self.env.action_space

        # observations for controller is a sliding window of 4 observations
        self.observation_space = spaces.Box(-1.0,1.0,(self.mem_len,52), dtype=float)

        #defuault observation is 4 lots of nothing
        self.observation = np.zeros((self.mem_len,52))

        self.action = None
        self.env = self.BLenv

    # reset doesnt reset the sliding window of the agent so it can differentiate between
    # agents across episode boundaries
    def reset(self):
        self.steps = 0
        #rest the environments of each attacker
        self.BLenv.reset()
        self.RMenv.reset()
        #if random.choice([0,1]) == 0:
        #    self.env = self.BLenv
        #else:
        #    self.env = self.RMenv
        self.env.reset()
        return np.zeros((self.mem_len,52))

    def sample_tasks(self, n_tasks):
        # Samples a goal position (2x1 position ector)
        #a = np.random.random(n_tasks) * 2 * np.pi
        #r = 3 * np.random.random(n_tasks) ** 0.5
        return np.random.choice((-1.0, 1.0), (n_tasks,))
        #return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        if task == -1:
            self.env = self.BLenv
            self.adversary = -1.0
        elif task == 1:
            self.adversary = 1.0
            self.env = self.RMenv
        else:
            print('error')

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.adversary

    def step(self, action=None):
        # select agent
        observation, reward, done, info = self.env.step(action)

        # update sliding window
        self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        self.observation[self.mem_len-1] = observation      # Replace what's on the rightmost position

        self.steps += 1
        if self.steps == self.max_steps:
            return self.observation, reward, True, info
        assert(self.steps <= self.max_steps)
        result = self.observation, reward, done, info
        return result

    def seed(self, seed=None):
        random.seed(seed)
