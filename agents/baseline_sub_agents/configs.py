from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.impala.impala import DEFAULT_CONFIG as IMPALA_CONFIG
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG
from ray.rllib.agents.sac.sac import DEFAULT_CONFIG as SAC_CONFIG
from ray.rllib.agents.maml.maml import DEFAULT_CONFIG as MAML_CONFIG

from ray.rllib.agents.trainer import Trainer
from CybORGAgent import *
import os

APEX_config = Trainer.merge_trainer_configs(
        APEX_DEFAULT_CONFIG,{
        "env": CybORGAgent,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_APEX_Model",
            # "vf_share_layers": False,
        },
        "lr": 0.0005,
        #"momentum": tune.uniform(0, 1),
        "num_workers": 2,  # parallelism
        "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
        #"vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        #"clip_param": 0.5,
        #"vf_clip_param": 5.0,

    })

APEX_RAINBOW_config = Trainer.merge_trainer_configs(
    APEX_DEFAULT_CONFIG,
    {
        "env": CybORGAgent,

        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "model": {
            "custom_model": "CybORG_APEX_Model",
            "vf_share_layers": True,
        },

        "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)

        # === Settings for Rollout Worker processes ===
        "num_workers": 3,  # No. rollout workers for parallel sampling.

        # === Settings for the Trainer process ===
        "lr": 1e-4,

        "learning_starts": 512,  # Number of steps of the evvironment to collect before learing starts
        "buffer_size": 1000000,
        "train_batch_size": 512,
        "rollout_fragment_length": 50,
        "target_network_update_freq": 250000,
        "timesteps_per_iteration": 12500,

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

    }
)

IMPALA_config = Trainer.merge_trainer_configs(
    IMPALA_CONFIG,
    {
        "env": CybORGAgent,

        # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_IMPALA_Model",
            "vf_share_layers": False,
        },
        "lr": 0.0005,
        # "momentum": tune.uniform(0, 1),
        "num_workers": 2,  # parallelism
        "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
        #"vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        #"clip_param": 0.5,
        #"vf_clip_param": 5.0,
    }
)
A2C_config = Trainer.merge_trainer_configs(
    A2C_DEFAULT_CONFIG,
    {
        "env": CybORGAgent,

        # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_A2C_Model",
            "vf_share_layers": False,
        },
        "lr": 0.0005,
        # "momentum": tune.uniform(0, 1),
        "num_workers": 2,  # parallelism
        "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
        # "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        # "clip_param": 0.5,
        # "vf_clip_param": 5.0,
    }
)

SAC_config = Trainer.merge_trainer_configs(
    SAC_CONFIG,
    {
        "env": CybORGAgent,

        # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "Q_model": {
            "custom_model": "CybORG_SAC_Model",
            # "vf_share_layers": False,
        },
        "policy_model": {
            "custom_model": "CybORG_SAC_Model",
            # "vf_share_layers": False,
        },
        "lr": 0.0005,
        # "momentum": tune.uniform(0, 1),
        "num_workers": 2,  # parallelism
        "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
        # "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        # "clip_param": 0.5,
        # "vf_clip_param": 5.0,
    }
)

#has to use the CybORGMAMLWrapped environment to run correctly
MAML_config = Trainer.merge_trainer_configs(
    MAML_CONFIG,
    {
        "env": CybORGAgent,

        # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_MAML_Model",
            # "vf_share_layers": False,
        },
        "lr": 0.0005,
        # "momentum": tune.uniform(0, 1),
        "num_workers": 2,  # parallelism
        "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
        # "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        # "clip_param": 0.5,
        # "vf_clip_param": 5.0,
    }
)