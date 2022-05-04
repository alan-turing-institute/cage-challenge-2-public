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
import random
from sub_agents import sub_agents
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.rllib.env import MultiAgentEnv

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


class FeudalEnv(MultiAgentEnv):
    # Env parameters
    max_steps = 100 # Careful! There are two other envs!
    mem_len = 1

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        super().__init__()
        self._skip_env_checking = True
        #self.env = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.cyborg = CybORG(self.path, 'sim', agents={'Red':RedMeanderAgent})
        self.RMenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.cyborg = CybORG(self.path, 'sim', agents={'Red':B_lineAgent})
        self.BLenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.lower_env = self.BLenv
        self.higher_env = self.lower_env
        self.steps = 0
        self.agent_name = 'BlueFeudal'
        self.mem_len = 1
        #action space is 2 for each trained agent to select from
        self.action_space = self.lower_env.action_space

        # observations for controller is a sliding window of 4 observations
        #self.observation_space = spaces.Box(-1.0,1.0,(self.mem_len,52), dtype=float)

        #defuault observation is 4 lots of nothing
        self.observation = np.zeros((self.mem_len,52), dtype=np.float32)

        self.action = None
        self.current_high_goal = None
        self.current_mid_goal = None

    # reset doesnt reset the sliding window of the agent so it can differentiate between
    # agents across episode boundaries
    def reset(self):
        self.steps = 0
        #rest the environments of each attacker
        self.BLenv.reset()
        self.current_high_goal = None
        self.current_mid_goal = None
        self.RMenv.reset()
        self.steps_remaining_at_high_level = None
        self.steps_remaining_at_mid_level = None
        self.num_high_level_steps = 0
        self.num_mid_level_steps = 0
        # current low level agent id. This must be unique for each high level
        # step since agent ids cannot be reused.
        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        self.mid_level_agent_id = "mid_level_{}".format(self.num_high_level_steps)
        if random.choice([0,1]) == 0:
            self.lower_env = self.BLenv
        else:
            self.lower_env = self.RMenv
        #self.higher_env = self.lower_env
        self.observation = np.zeros((self.mem_len,52), dtype=np.float32)
        return {"high_level_agent": self.observation}

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "high_level_agent" in action_dict:
            return self._high_level_step(action_dict["high_level_agent"])
        elif "mid_level_" in list(action_dict)[0]:
            return self._mid_level_step(action_dict[list(action_dict)[0]])
        elif "low_level_" in list(action_dict)[0]:
            return self._low_level_step(list(action_dict.values())[0])
        else:
            print(action_dict)
            print('agent level not possible')
            exit(2)

    def _high_level_step(self, action):
        #logger.debug("High level agent sets goal")
        self.current_high_goal = action
        self.steps_remaining_at_high_level = 1
        self.num_high_level_steps += 1
        self.mid_level_agent_id = "mid_level_{}".format(self.num_high_level_steps)
        obs = {self.mid_level_agent_id: [self.observation, self.current_high_goal]}
        rew = {self.mid_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def _mid_level_step(self, action):
        self.current_mid_goal = action
        #logger.debug("High level agent sets goal")
        self.steps_remaining_at_mid_level = 1
        self.num_mid_level_steps += 1
        self.steps_remaining_at_high_level -= 1
        self.low_level_agent_id = "low_level_{}".format(self.num_mid_level_steps)
        obs = {self.low_level_agent_id: [self.observation, self.current_high_goal, self.current_mid_goal]}
        rew = {self.low_level_agent_id: 0}
        if self.current_mid_goal == 0 and 'Analyse' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'Remove' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'Analyse' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'Remove' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'Misinform' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'DecoyApache' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'DecoyFemitter' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'DecoyHarakaSMPT' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'DecorSmss' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'DecoySSHD' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'DecoySvchost' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'DecoyTomcat' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'DecoyVsftpd' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        elif self.current_mid_goal == 0 and 'Restore' in str(self.lower_env.get_last_action('Blue')):
            rew[self.mid_level_agent_id] = 0
        else:
            rew[self.mid_level_agent_id] = -1
        done = {"__all__": False}
        return obs, rew, done, {}

    def _low_level_step(self, action):
        #logger.debug("Low level agent step {}".format(action))
        self.higher_env = self.lower_env
        self.steps_remaining_at_mid_level -= 1
        #cur_pos = tuple(self.observation[0])

        #h_obs, h_rew, h_done, h_info = self.higher_env.step(self.current_high_goal)

        # Step in the actual env
        l_obs, l_rew, l_done, l_info = self.lower_env.step(action)
        #new_pos = tuple(f_obs[0])
        l_obs = np.reshape(l_obs, (1, 52))
        self.observation = l_obs

        # Calculate low-level agent observation and reward
        rew = {}
        obs = {self.low_level_agent_id: [l_obs, self.current_high_goal]}
        if self.current_high_goal == 0 and (action == 0 or action == 1):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 0 and 'Defender' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 1 and 'Enterprise0' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 2 and 'Enterprise1' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 3 and 'Enterprise2' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 4 and 'Op_Host0' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 5 and 'Op_Host1' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 6 and 'Op_Host2' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 7 and 'Op_Server0' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 8 and 'User0' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 9 and 'User1' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 10 and 'User2' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 11 and 'User3' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        elif self.current_high_goal == 12 and 'User4' in str(self.lower_env.get_last_action('Blue')):
            rew[self.low_level_agent_id] = 0
        else:
            rew[self.low_level_agent_id] = -1


        #if np.array_equal(l_obs, h_obs):
        #    rew = {self.low_level_agent_id: 1}
        #else:
        #    rew = {self.low_level_agent_id: -1}

        # Handle env termination & transitions back to higher level
        self.steps += 1
        if self.steps == self.max_steps:
            l_done = True
        done = {"__all__": False}
        if l_done:
            done["__all__"] = True
            #logger.debug("high level final reward {}".format(f_rew))
            rew["high_level_agent"] = l_rew
            obs["high_level_agent"] = l_obs
        elif self.steps_remaining_at_high_level == 0:
            done[self.low_level_agent_id] = True
            rew["high_level_agent"] = l_rew
            obs["high_level_agent"] = l_obs

        return obs, rew, done, {}


    def seed(self, seed=None):
        random.seed(seed)
