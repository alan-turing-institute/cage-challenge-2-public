import numpy as np
import pickle as pkl
from neural_nets import *
import os
from sub_agents import *
import os.path as path
from CybORG.Agents import B_lineAgent, SleepAgent, RedMeanderAgent
from configs import *
# from CybORGActionAgent import CybORGActionAgent
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from bline_CybORGAgent import CybORGAgent as bline_CybORGAgent

class LoadBanditBlueAgent:

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_hier_Model", TorchModel)

        # Load checkpoint locations of each agent
        two_up = path.abspath(path.join(__file__, "../../../"))
        self.CTRL_checkpoint_pointer = two_up + sub_agents['bandit_trained']
        self.BL_checkpoint_pointer = two_up + sub_agents['B_line_trained']
        self.RM_checkpoint_pointer = two_up + sub_agents['RedMeander_trained']

        print("Using checkpoint file (Controller): {}".format(self.CTRL_checkpoint_pointer))
        print("Using checkpoint file (B-line): {}".format(self.BL_checkpoint_pointer))
        print("Using checkpoint file (Red Meander): {}".format(self.RM_checkpoint_pointer))

        # Restore the controller model
        with open(self.CTRL_checkpoint_pointer, "rb") as controller_chkpt:  # Must open file in binary mode for pickle
            self.controller = pkl.load(controller_chkpt)

        self.bandit_observation = np.array([], dtype=int)
        RM_config = meander_config
        RM_config["in_evaluation"] = True
        RM_config["explore"] = False

        BL_config = bline_config
        BL_config["in_evaluation"] = True
        BL_config["explore"] = False
        
        #load agent trained against RedMeanderAgent
        self.RM_def = PPOTrainer(config=RM_config, env=CybORGAgent)
        self.RM_def.restore(self.RM_checkpoint_pointer)

        #load agent trained against B_lineAgent
        BL_config['env'] = bline_CybORGAgent
        BL_config["env_config"] = {'agent_name': 'Blue', 'env': None, 'max_steps': 100, 'attacker': B_lineAgent}
        self.BL_def = ppo.PPOTrainer(config=BL_config, env=bline_CybORGAgent)
        self.BL_def.restore(self.BL_checkpoint_pointer)

        #self.red_agent=-1
        self.state = [np.zeros(256, np.float32),
                      np.zeros(256, np.float32)]
        self.step = 0
        # heuristics
        self.set = False
        self.observations = []
        self.adversary = 0


    def set_red_agent(self, red_agent):
        self.red_agent = red_agent

    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        #update sliding window
        # discover network services sequence
        self.step += 1
        if self.step < 5:
            self.bandit_observation = np.append(self.bandit_observation, obs[2:])
            #return 0, -1
        elif self.step == 5:
            bandit_obs_hashable = ''.join(str(bit) for bit in self.bandit_observation)
            self.adversary = np.argmax(self.controller[bandit_obs_hashable])


        #select agent to compute action
        """ if self.red_agent == B_lineAgent or self.red_agent == SleepAgent:
            agent_to_select = 0
        else: #RedMeanderAgent
            agent_to_select = 1"""
        #agent_to_select = self.controller_agent.compute_single_action(obs)
        if self.adversary == 0:
            #print('b_line defence')
            # get action from agent trained against the B_lineAgent

            # keep track of the lstm state for later use
            _, self.state, _ = self.RM_def.compute_single_action(obs[2:], self.state)
            bl_obs = self.bits_to_float(obs)
            agent_action = self.BL_def.compute_single_action(bl_obs)
        elif self.adversary == 1:
            # get action from agent trained against the RedMeanderAgent
            agent_action, state, _ = self.RM_def.compute_single_action(obs[2:], self.state)
            #print('meander defence')
            self.state = state
            # agent_action = self.RM_def.compute_single_action(self.observation[-1:])
        elif self.adversary == 2:
            agent_action = 0

        else:
            print('something went terribly wrong, old sport')
        return agent_action, self.adversary

    def end_episode(self):
        self.set = False
        self.state = [np.zeros(256, np.float32),
               np.zeros(256, np.float32)]
        self.bandit_observation = np.array([], dtype=int)
        #self.observations = []
        self.adversary = 0
        self.step = 0

    def bits_to_float(self, obs):
        float_list = []

        rest_obs = np.reshape(obs[2:], (13, 4))
        for host in rest_obs:
            activity = np.array(host[:2])
            compromised = np.array(host[2:])
            if all(activity == [0,0]):
                value = [0.]
            elif all(activity == [1,0]):
                value = [1.]
            elif all(activity == [1,1]):
                value = [2.]
            else: 
                raise ValueError('not activity type')
            float_list += value

            # Compromised
            if all(compromised == [0, 0]):
                value = [0.]
            elif all(compromised == [1, 0]):
                value = [1.]
            elif all(compromised == [0,1]):
                value = [2.]
            elif all(compromised == [1,1]):
                value = [3.]
            else: 
                raise ValueError('not compromised type')
            float_list += value

        success = obs[:2]
        if all(success == [1, 0]):
            float_list += [0.]
        elif all(success == [0, 0]):
            float_list += [1.]
        elif all(success == [0, 1]):
            float_list += [2.]

        return np.array(float_list)
        
        