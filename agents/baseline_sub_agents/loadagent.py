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
from CybORGAgent import CybORGAgent


class LoadBlueAgent:
    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """

    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_APEX_Model", TorchModel)

        #with open("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        self.checkpoint_pointer = '../../logs/various/APEX_B_lineAgent_2022-04-29_18-14-12/APEX_CybORGAgent_c9eaf_00000_0_2022-04-29_18-14-12/checkpoint_000007/checkpoint-7'
        print("Using checkpoint file: {}".format(self.checkpoint_pointer))

        config = APEX_RAINBOW_config

        # Restore the checkpointed model
        self.agent = dqn.ApexTrainer(config=config, env=CybORGAgent)
        self.agent.restore(self.checkpoint_pointer)

    """Compensate for the different method name"""

    def get_action(self, obs, action_space):
        return self.agent.compute_single_action(obs)