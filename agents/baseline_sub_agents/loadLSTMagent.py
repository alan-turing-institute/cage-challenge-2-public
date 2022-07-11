import os
from pprint import pprint

import ray
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.maml import DEFAULT_CONFIG as MAML_CONFIG
import ray.rllib.agents.a3c as a2c
from configs import *
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.impala as impala
import ray.rllib.agents.dqn as dqn

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from train_algos import TorchModel
#from CybORGAgent import CybORGAgent
from ray.rllib.models.tf.attention_net import GTrXLNet
from configs import *
from neural_nets import *

class LoadBlueAgent:
    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """

    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_Torch", TorchModel)
        ModelCatalog.register_custom_model("CybORG_GTrXL_Model", GTrXLNet)

        #with open("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        self.checkpoint_pointer = '/Users/mylesfoley/Desktop/Imperial/git/turing/cage-challenge-2/logs/various/PPO_LSTM_RedMeanderAgent_2022-07-06_16-32-36/PPO_CybORGAgent_dcaaa_00000_0_2022-07-06_16-32-36/checkpoint_001829/checkpoint-1829'
        print("Using checkpoint file: {}".format(self.checkpoint_pointer))


        config = LSTM_config
        config["in_evaluation"] = True
        config["explore"] = False
        # Restore the checkpointed model
        self.agent = ppo.PPOTrainer(config=config, env=CybORGAgent)
        self.agent.restore(self.checkpoint_pointer)
        self.state=[np.zeros(256, np.float32),
               np.zeros(256, np.float32)]
        self.prev_action = 0
        self.prev_reward = 0


    """Compensate for the different method name"""

    def get_action(self, obs, action_space):
        #self.agent.compute_action(obs)
        action, state, logits = self.agent.compute_single_action(obs, self.state)
        self.state = state

        return action

    def end_episode(self):
        self.state = [np.zeros(256, np.float32),
               np.zeros(256, np.float32)]