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
from ray.rllib.models.tf.attention_net import GTrXLNet
from configs import *
from neural_nets import *
from CybORGMultiAdversaryAgent import CybORGMultiAgent

class LoadBlueAgent:
    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """

    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_Torch", TorchModel)
        ModelCatalog.register_custom_model("CybORG_GTrXL_Model", GTrXLNet)

        #with open("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        self.checkpoint_pointer = '/Users/mylesfoley/Desktop/Imperial/git/turing/cage-challenge-2/logs/various/PPO_curiosity_3_layer_B_lineAgent_2022-06-06_12-46-35/PPO_CybORGAgent_51413_00000_0_2022-06-06_12-46-35/checkpoint_001018/checkpoint-1018'
        print("Using checkpoint file: {}".format(self.checkpoint_pointer))

        """config = {
            "env": CybORGAgent,
            # This env_config is only used for the RepeatAfterMeEnv env.
            "env_config": {
                "repeat_delay": 2,
            },
            "gamma": 0.99,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
            "num_envs_per_worker": 20,
            "entropy_coeff": 0.001,
            "num_sgd_iter": 10,
            "vf_loss_coeff": 1e-5,
            "model": {
                # Attention net wrapping (for tf) can already use the native keras
                # model versions. For torch, this will have no effect.
                "_use_default_native_models": True,
                "use_attention": not True,
                "max_seq_len": 10,
                "attention_num_transformer_units": 1,
                "attention_dim": 32,
                "attention_memory_inference": 10,
                "attention_memory_training": 10,
                "attention_num_heads": 1,
                "attention_head_dim": 32,
                "attention_position_wise_mlp_dim": 32,
            },
            "framework": 'torch',
        }"""
        config = PPO_Curiosity_config
        config["in_evaluation"] = True
        config["explore"] = False
        # Restore the checkpointed model
        self.agent = ppo.PPOTrainer(config=config, env=CybORGAgent)
        self.agent.restore(self.checkpoint_pointer)

    """Compensate for the different method name"""

    def get_action(self, obs, action_space):
        return self.agent.compute_single_action(obs)

    def end_episode(self):
        pass