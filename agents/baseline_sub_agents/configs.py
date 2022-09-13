from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.impala.impala import DEFAULT_CONFIG as IMPALA_CONFIG
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG
from ray.rllib.agents.sac.sac import DEFAULT_CONFIG as SAC_CONFIG
from ray.rllib.agents.maml.maml import DEFAULT_CONFIG as MAML_CONFIG
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.trainer import Trainer
from CybORGAgent import *
#from CybORGMAML import *
import os
from neural_nets import *
from ray.rllib.models.tf.attention_net import GTrXLNet
from ray import tune
ModelCatalog.register_custom_model("CybORG_Torch", TorchModel)
ModelCatalog.register_custom_model("CybORG_TF", TFModel)
ModelCatalog.register_custom_model("CybORG_GTrXL_Model", GTrXLNet)
from curiosity import Curiosity

# APEX_config = Trainer.merge_trainer_configs(
#         APEX_DEFAULT_CONFIG,{
#         "env": CybORGAgent,
#         "env_config": {
#             "null": 0,
#         },
#         # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
#         "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
#         "model": {
#             "custom_model": "CybORG_Torch",
#             # "vf_share_layers": False,
#         },
#         "lr": 0.0005,
#         #"momentum": tune.uniform(0, 1),
#         "num_workers": 2,  # parallelism
#         "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
#         "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
#         #"vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
#         #"clip_param": 0.5,
#         #"vf_clip_param": 5.0,

#     })

# APEX_RAINBOW_config = Trainer.merge_trainer_configs(
#     APEX_DEFAULT_CONFIG,
#     {
#         "env": CybORGAgent,

#         "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#         "model": {
#             "custom_model": "CybORG_Torch",
#             "vf_share_layers": True,
#         },

#         "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
#         "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)

#         # === Settings for Rollout Worker processes ===
#         "num_workers": 3,  # No. rollout workers for parallel sampling.

#         # === Settings for the Trainer process ===
#         "lr": 1e-4,

#         "learning_starts": 512,  # Number of steps of the evvironment to collect before learing starts
#         "buffer_size": 1000000,
#         "train_batch_size": 512,
#         "rollout_fragment_length": 50,
#         "target_network_update_freq": 250000,
#         "timesteps_per_iteration": 12500,

#         # === Environment settings ===
#         # "preprocessor_pref": "deepmind",

#         # === DQN/Rainbow Model subset config ===
#         "num_atoms": 1,  # Number of atoms for representing the distribution of return.
#         # Use >1 for distributional Q-learning (Rainbow config)
#         # 1 improves faster than 2
#         "v_min": -1000.0,  # Minimum Score
#         "v_max": -0.0,  # Set to maximum score
#         "noisy": True,  # Whether to use noisy network (Set True for Rainbow)
#         "sigma0": 0.5,  # control the initial value of noisy nets
#         "dueling": True,  # Whether to use dueling dqn
#         "hiddens": [256],  # Dense-layer setup for each the advantage branch and the value
#         # branch in a dueling architecture.
#         "double_q": True,  # Whether to use double dqn
#         "n_step": 3,  # N-step Q learning (Out of 1, 3 and 6, 3 seems to do learn most quickly)

#     }
# )

# IMPALA_config = Trainer.merge_trainer_configs(
#     IMPALA_CONFIG,
#     {
#         "env": CybORGAgent,

#         # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
#         "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
#         "model": {
#             "custom_model": "CybORG_Torch",
#             "vf_share_layers": False,
#         },
#         "lr": 0.0005,
#         # "momentum": tune.uniform(0, 1),
#         "num_workers": 2,  # parallelism
#         "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
#         "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
#         #"vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
#         #"clip_param": 0.5,
#         #"vf_clip_param": 5.0,
#     }
# )
# A2C_config = Trainer.merge_trainer_configs(
#     A2C_DEFAULT_CONFIG,
#     {
#         "env": CybORGAgent,

#         # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
#         "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
#         "model": {
#             "custom_model": "CybORG_Torch",
#             "vf_share_layers": False,
#         },
#         "lr": 0.0005,
#         # "momentum": tune.uniform(0, 1),
#         "num_workers": 2,  # parallelism
#         "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
#         "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
#         # "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
#         # "clip_param": 0.5,
#         # "vf_clip_param": 5.0,
#     }
# )

# SAC_config = Trainer.merge_trainer_configs(
#     SAC_CONFIG,
#     {
#         "env": CybORGAgent,

#         # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
#         "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
#         "Q_model": {
#             "custom_model": "CybORG_Torch",
#             # "vf_share_layers": False,
#         },
#         "policy_model": {
#             "custom_model": "CybORG_Torch",
#             # "vf_share_layers": False,
#         },
#         "lr": 0.0005,
#         # "momentum": tune.uniform(0, 1),
#         "num_workers": 2,  # parallelism
#         "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
#         "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
#         # "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
#         # "clip_param": 0.5,
#         # "vf_clip_param": 5.0,
#     }
# )


# GTrXL_config = Trainer.merge_trainer_configs(
#     PPO_CONFIG,
#     {
#         "env": CybORGAgent,
#         # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
#         "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
#         "num_workers": 0,
#         "gamma": 0.99,
#         "num_envs_per_worker": 20,
#         "entropy_coeff": 0.001,
#         "vf_loss_coeff": 1e-5,
#         "num_sgd_iter": 10,
#         "model": {
#             "custom_model": "CybORG_GTrXL_Model",
#             "max_seq_len": 10,
#             "custom_model_config": {
#                 "num_transformer_units": 1,
#                 "attention_dim": 32,
#                 "num_heads": 1,
#                 "memory_inference": 5,
#                 "memory_training": 5,
#                 "head_dim": 32,
#                 "position_wise_mlp_dim": 32,
#             },
#         },
#         # "lr": 0.0005,
#         # "momentum": tune.uniform(0, 1),
#         # "num_workers": 2,  # parallelism
#         "framework": "tf",  # May also use "tf2", "tfe" or "torch" if supported
#         # "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
#         # "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
#         # "clip_param": 0.5,
#         # "vf_clip_param": 5.0,
#     }
# )

# attention_config = {
#     "env": CybORGAgent,
#     "env_config": {
#         "null": 0,
#     },
#     "gamma": 0.99,
#     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
#     "num_envs_per_worker": 20,
#     "entropy_coeff": 0.001,
#     "num_sgd_iter": 10,
#     "vf_loss_coeff": 1e-5,
#     "model": {
#         # Attention net wrapping (for tf) can already use the native keras
#         # model versions. For torch, this will have no effect.
#         "_use_default_native_models": True,
#         #"custom_model": "CybORG_Torch",
#         #'fcnet_hiddens': [256, 256, 52],
#         "use_attention": True,
#         "use_lstm":  not True,
#         "max_seq_len": 10,
#         "attention_num_transformer_units": 1,
#         "attention_dim": 32,
#         "attention_memory_inference": 10,
#         "attention_memory_training": 10,
#         "attention_num_heads": 1,
#         "attention_head_dim": 32,
#         "attention_position_wise_mlp_dim": 32,
#     },
#     "framework": 'tf',
# }

meander_config = {
    "env": CybORGAgent,
    "env_config": {
        "null": 0,
    },
    "gamma": 0.99,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
    "num_envs_per_worker": 20,
    "entropy_coeff": 0.001,
    "num_sgd_iter": 10,
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



# LSTM_curiosity_config = {
#     "env": CybORGAgent,
#     "env_config": {
#         "null": 0,
#     },
#     "gamma": 0.99,
#     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
#     "num_envs_per_worker": 20,
#     "entropy_coeff": 0.001,
#     "num_sgd_iter": 10,
#     #"vf_loss_coeff": 1e-5,
#     #"vf_share_layers": False,
#     "model": {
#         # Attention net wrapping (for tf) can already use the native keras
#         # model versions. For torch, this will have no effect.
#         "_use_default_native_models": True,
#         "custom_model": "CybORG_Torch",
#         'fcnet_hiddens': [256, 256, 52],
#         "use_attention": not True,
#         "use_lstm":  not True,
#         "max_seq_len": 10,
#         "lstm_use_prev_action": True,
#         "lstm_use_prev_reward": True,

#     },
#     "framework": 'torch',
#     "exploration_config": {
#         "type": "Curiosity",  # <- Use the Curiosity module for exploring.
#         "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
#         "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
#         "feature_dim": 53,  # Dimensionality of the generated feature vectors.
#         # Setup of the feature net (used to encode observations into feature (latent) vectors).
#         "feature_net_config": {
#             "fcnet_hiddens": [],
#             "fcnet_activation": "relu",
#         },
#         "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
#         "inverse_net_activation": "relu",  # Activation of the "inverse" model.
#         "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
#         "forward_net_activation": "relu",  # Activation of the "forward" model.
#         "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
#         # Specify, which exploration sub-type to use (usually, the algo's "default"
#         # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
#         "sub_exploration": {
#             "type": "StochasticSampling",
#         }
#     }
# }

# PPO_Curiosity_config = Trainer.merge_trainer_configs(
#     PPO_CONFIG, {
#         "env": CybORGAgent,
#         "env_config": {
#             "null": 0,
#         },
#         # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#         "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
#         "model": {
#             "custom_model": "CybORG_Torch",
#             'fcnet_hiddens': [256, 256],
#             "vf_share_layers": False,
#         },
#         "lr": 0.0005,
#         # "momentum": tune.uniform(0, 1),
#         "num_workers": 0,  # parallelism
#         "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
#         "eager_tracing": True,
#         # In order to reach similar execution speed as with static-graph mode (tf default)
#         "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
#         "clip_param": 0.5,

#         "vf_clip_param": 5.0,
#         "exploration_config": {
#             "type": "Curiosity",  # <- Use the Curiosity module for exploring.
#             "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
#             "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
#             "feature_dim": 53,  # Dimensionality of the generated feature vectors.
#             # Setup of the feature net (used to encode observations into feature (latent) vectors).
#             "feature_net_config": {
#                 "fcnet_hiddens": [],
#                 "fcnet_activation": "relu",
#             },
#             "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
#             "inverse_net_activation": "relu",  # Activation of the "inverse" model.
#             "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
#             "forward_net_activation": "relu",  # Activation of the "forward" model.
#             "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
#             # Specify, which exploration sub-type to use (usually, the algo's "default"
#             # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
#             "sub_exploration": {
#                 "type": "StochasticSampling",
#             }
#         }
#     })

# small_nn_config = {
#     "env": CybORGAgent,
#     "env_config": {
#         "null": 0,
#     },
#     "gamma": 0.99,
#     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
#     "num_envs_per_worker": 20,
#     "entropy_coeff": 0.001,
#     "num_sgd_iter": 10,
#     #"vf_loss_coeff": 1e-5,
#     #"vf_share_layers": False,
#     "model": {
#         # Attention net wrapping (for tf) can already use the native keras
#         # model versions. For torch, this will have no effect.
#         "_use_default_native_models": True,
#         "custom_model": "CybORG_Torch",
#         'fcnet_hiddens': [104, 104, 52],
#         "use_attention": not True,

#     },
#     "framework": 'torch',
# }

bline_config = Trainer.merge_trainer_configs(
        PPO_CONFIG,{
        "env": CybORGAgent,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            # ## autoreg 1
            # "custom_model": "autoregressive_model",
            # "custom_action_dist": "binary_autoreg_dist",
            ## autoreg 2 (similar to original)
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