import os
from pprint import pprint

import ray
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from train_rllib_alt import CybORGAgent, CustomModel


class LoadBlueAgent:
    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """

    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_DQN_Model", CustomModel)

        with open("checkpoint_pointer.txt", "r") as chkpopfile:
            self.checkpoint_pointer = chkpopfile.readlines()[0]
        print("Using checkpoint file: {}".format(self.checkpoint_pointer))

        config = Trainer.merge_trainer_configs(
            APEX_DEFAULT_CONFIG,
            {
                "env": CybORGAgent,

                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
                "model": {
                    "custom_model": "CybORG_DQN_Model",
                    "vf_share_layers": True,
                },

                "framework": "tf2",  # May also use "tf2", "tfe" or "torch" if supported
                "eager_tracing": True,
                # In order to reach similar execution speed as with static-graph mode (tf default)

                # === Settings for Rollout Worker processes ===
                "num_workers": 4,  # No. rollout workers for parallel sampling.

                # === Settings for the Trainer process ===
                "lr": 1e-4,

                # === Environment settings ===
                # "preprocessor_pref": "deepmind",

                # === DQN/Rainbow Model subset config ===
                "num_atoms": 1,  # Number of atoms for representing the distribution of return.
                # Use >1 for distributional Q-learning (Rainbow config)
                # 1 improves faster than 2
                "v_min": -1000.0,  # Minimum Score
                "v_max": -0.0,  # Set to maximum score
                "noisy": True,  # Whether to use noisy network (Set True for Rainbow)
                "sigma0": 0.5,  # control the initial value of noisy nets
                "dueling": True,  # Whether to use dueling dqn
                "hiddens": [256],  # Dense-layer setup for each the advantage branch and the value
                # branch in a dueling architecture.
                "double_q": True,  # Whether to use double dqn
                "n_step": 3,  # N-step Q learning (Out of 1, 3 and 6, 3 seems to do learn most quickly)

                "learning_starts": 100,  # Number of steps of the evvironment to collect before learing starts

            }
        )

        # Restore the checkpointed model
        self.agent = dqn.ApexTrainer(config=config, env=CybORGAgent)
        self.agent.restore(self.checkpoint_pointer)

    """Compensate for the different method name"""

    def get_action(self, obs, action_space):
        return self.agent.compute_single_action(obs)