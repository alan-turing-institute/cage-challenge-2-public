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
from configs import *
from neural_nets import *

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


class HierEnv(gym.Env):
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


        #relative_path = #'cage-challenge-1' #[:62], os.path.abspath(os.getcwd()) +
        #print(relative_path)
        two_up = path.abspath(path.join(__file__, "../../.."))

        self.BL_checkpoint_pointer = two_up +  sub_agents['B_line_trained']
        self.RM_checkpoint_pointer = two_up + sub_agents['RedMeander_trained']

        """RM_config  = {
        "env": CybORGAgent,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_Torch",
            'fcnet_hiddens': [256, 256, 52],
            "vf_share_layers": False,
            'use_lstm': True,
        },
        "lr": 0.0005,
        # "momentum": tune.uniform(0, 1),
        "num_workers": 0,  # parallelism
        "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True,
        # In order to reach similar execution speed as with static-graph mode (tf default)
        "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        "clip_param": 0.5,

        "vf_clip_param": 5.0,

        }"""
        RM_config = LSTM_config
        RM_config["num_workers"] = 1
        RM_config["in_evaluation"] = True
        RM_config["explore"] = False
        RM_config["create_env_on_driver"] = False
        RM_config['num_workers'] = 0

        BL_config = PPO_Curiosity_config
        BL_config['model']['fcnet_hiddens'] = [256, 256, 256]
        BL_config["in_evaluation"] = True
        BL_config["explore"] = False

        ModelCatalog.register_custom_model("CybORG_Torch", TorchModel)
        ModelCatalog.register_custom_model("CybORG_GTrXL_Model", GTrXLNet)
        # Restore the checkpointed model
        self.BL_def = ppo.PPOTrainer(config=BL_config, env=CybORGAgent)
        print('loaded BL')
        self.RM_def = ppo.PPOTrainer(config=RM_config, env=CybORGAgent)
        #sub_config['env'] = CybORGAgent
        print('initalising agents...')
        self.RM_def.load_checkpoint(self.RM_checkpoint_pointer)
        self.RM_def.restore(self.RM_checkpoint_pointer)
        self.BL_def.restore(self.BL_checkpoint_pointer)

        self.steps = 0
        self.agent_name = 'BlueHier'

        #action space is 2 for each trained agent to select from
        self.action_space = spaces.Discrete(2)

        # observations for controller is a sliding window of 4 observations
        self.observation_space = spaces.Box(-1.0,1.0,(self.mem_len,52), dtype=float)

        #defuault observation is 4 lots of nothing
        self.observation = np.zeros((self.mem_len,52))

        self.state = [np.zeros(256, np.float32),
                      np.zeros(256, np.float32)]

        self.action = None
        self.env = self.BLenv
        self.adversary = 'B_line'
        print('Hier env set up')

    # reset doesnt reset the sliding window of the agent so it can differentiate between
    # agents across episode boundaries
    def reset(self):
        self.steps = 0
        #rest the environments of each attacker
        self.BLenv.reset()
        self.RMenv.reset()
        self.state=[np.zeros(256, np.float32),
               np.zeros(256, np.float32)]
        if random.choice([0,1]) == 0:
            self.env = self.BLenv
        else:
            self.env = self.RMenv
        return np.zeros((self.mem_len,52))

    def step(self, action=None):
        # select agent
        if action == 0:
            # get action from agent trained against the B_lineAgent
            agent_action = self.BL_def.compute_single_action(self.observation[-1:])

            #keep track of the lstm state for later use
            _, self.state, _ = self.RM_def.compute_single_action(self.observation[-1:], self.state)
        elif action == 1:
            # get action from agent trained against the RedMeanderAgent
            agent_action, self.state, _ = self.RM_def.compute_single_action(self.observation[-1:], self.state)
            #self.state = state
            #agent_action = self.RM_def.compute_single_action(self.observation[-1:])
        else:
            print('something went terribly wrong, old sport')
        observation, reward, done, info = self.env.step(agent_action)

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
