import numpy as np
import pickle as pkl
from neural_nets import *
import os
from sub_agents import *
import os.path as path
from CybORG.Agents import B_lineAgent, SleepAgent, RedMeanderAgent
from configs import *
from CybORGActionAgent import CybORGActionAgent
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
class LoadBanditBlueAgent:

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_hier_Model", TorchModel)
        #relative_path = os.path.abspath(os.getcwd())[:62] + '/cage-challenge-1'
        #print("Relative path:", relative_path)

        # Load checkpoint locations of each agent
        two_up = path.abspath(path.join(__file__, "../../../"))
        #self.CTRL_checkpoint_pointer = two_up + '/log_dir/rl_controller_scaff/PPO_HierEnv_1e996_00000_0_2022-01-27_13-43-33/checkpoint_000212/checkpoint-212'
        self.CTRL_checkpoint_pointer = two_up + '/logs/bandits/controller_bandit_2022-07-15_11-08-56/bandit_controller_15000.pkl'
        self.BL_checkpoint_pointer = two_up + sub_agents['B_line_trained']
        self.RM_checkpoint_pointer = two_up + sub_agents['RedMeander_trained']

        #with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        print("Using checkpoint file (Controller): {}".format(self.CTRL_checkpoint_pointer))
        print("Using checkpoint file (B-line): {}".format(self.BL_checkpoint_pointer))
        print("Using checkpoint file (Red Meander): {}".format(self.RM_checkpoint_pointer))

        # Restore the controller model
        with open(self.CTRL_checkpoint_pointer, "rb") as controller_chkpt:  # Must open file in binary mode for pickle
            #print('Red Agent states loaded from {}'.format(controller_chpt))
            self.controller = pkl.load(controller_chkpt)

        self.bandit_observation = np.array([], dtype=int)
        RM_config = LSTM_config
        RM_config["in_evaluation"] = True
        RM_config["explore"] = False

        BL_config = PPO_Curiosity_config
        #BL_config['model']['fcnet_hiddens'] = [256, 256, 256]
        BL_config["in_evaluation"] = True
        BL_config["explore"] = False

        #load agent trained against RedMeanderAgent
        self.RM_def = ppo.PPOTrainer(config=RM_config, env=CybORGAgent)
        self.RM_def.restore(self.RM_checkpoint_pointer)
        #load agent trained against B_lineAgent
        BL_config['env'] = CybORGActionAgent
        BL_config["env_config"] = {'agent_name': 'Blue', 'env': None, 'max_steps': 100, 'attacker': B_lineAgent}
        self.BL_def = ppo.PPOTrainer(config=BL_config, env=CybORGActionAgent)
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
        #self.observations.append(obs)
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
            agent_action = self.BL_def.compute_single_action(obs)
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
