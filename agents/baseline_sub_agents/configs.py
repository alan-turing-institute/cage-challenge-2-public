from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.agents import Trainer
from CybORGAgent import CybORGAgent
from bline_CybORGAgent import CybORGAgent as bline_CybORGAgent
import os
from neural_nets import *
from ray import tune
ModelCatalog.register_custom_model("CybORG_Torch", TorchModel)
from curiosity import Curiosity


meander_config = {
    "env": CybORGAgent,
    "env_config": {
        "null": 0,
    },
    "gamma": 0.99,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
    "num_envs_per_worker": 4,
    "entropy_coeff": 0.001,
    "num_sgd_iter": 10,
    "horizon": 100,
    "rollout_fragment_length": 100,
    #"vf_loss_coeff": 1e-5,
    #"vf_share_layers": False,
    "model": {
        # Attention net wrapping (for tf) can already use the native keras
        # model versions. For torch, this will have no effect.
        "_use_default_native_models": True,
        "custom_model": "CybORG_Torch",
        'fcnet_hiddens': [256, 256, 52],
        "use_attention": not True,
        "use_lstm":  not True,
        "max_seq_len": 10,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,

    },
    "framework": 'torch',
}


bline_config = Trainer.merge_trainer_configs(
        PPO_CONFIG,{
        "env": bline_CybORGAgent,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_Torch",

            "vf_share_layers": False,
            "fcnet_hiddens": [256, 256],
        },
        "lr": 0.0005,
        #"momentum": tune.uniform(0, 1),
        "num_workers": 0,  # parallelism
        "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
        "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        "clip_param": 0.5,
        "vf_clip_param": 5.0,
        "exploration_config": {
            "type": Curiosity,  # <- Use the Curiosity module for exploring.
            "framework": "torch",
            "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 53,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            "feature_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
                'framework': 'torch',
                #'device': 'cuda:0'
            },
            "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            "sub_exploration": {
                "type": "StochasticSampling",
            }
        }
    })