"""Alternative RLLib model based on local training
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import inspect
import sys

# Ray imports
import ray
from ray import tune
from ray.tune import grid_search
from ray.tune.schedulers import ASHAScheduler # https://openreview.net/forum?id=S1Y7OOlRZ algo for early stopping
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import DEFAULT_CONFIG
import ray.rllib.agents.impala as impala
from ray.rllib.utils.framework import try_import_tf, try_import_torch
import time
from gym.spaces import Discrete, Tuple
# CybORG imports
from CybORG import CybORG
from CybORG.Agents.Wrappers import ChallengeWrapper
from typing import Any
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from CybORGFeudal import FeudalEnv


tf1, tf, tfv = try_import_tf()
path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
cyborg = ChallengeWrapper(env=CybORG(path, 'sim'), agent_name='Blue')
action_space = cyborg.action_space.n
obs_space = cyborg.observation_space

class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)

        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

def normc_initializer(std: float = 1.0) -> Any:
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(
            tensor.data.pow(2).sum(1, keepdim=True))

    return initializer

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

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id.startswith("low_level_"):
        return "low_level_policy"
    elif agent_id.startswith("mid_level_"):
        return "mid_level_policy"
    else:
        return "high_level_policy"

if __name__ == "__main__":
    ray.init()
    # Can also register the env creator function explicitly with register_env("env name", lambda config: EnvClass(config))
    ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)
    config = Trainer.merge_trainer_configs(
        DEFAULT_CONFIG,{
        "env": FeudalEnv,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_PPO_Model",
            "vf_share_layers": False,
        },
        "lr": 0.0005,
        "multiagent": {
            "policies": {
                # four actions for the top level agent
                "high_level_policy": (
                    None,
                    Box(-1.0, 1.0, (1, 52)),
                    Discrete(14),
                    {"gamma": 0.9},
                ),
                "mid_level_policy": (
                    None,
                    Tuple([Box(-1.0,1.0,(1, 52)), Discrete(14)]),
                    Discrete(14),
                    {"gamma": 0.9},
                ),
                "low_level_policy": (
                    None,
                    Tuple([Box(-1.0,1.0,(1, 52)), Discrete(14), Discrete(14)]),
                    FeudalEnv.action_space,
                    {"gamma": 0.0},
                ),
            },
            "policy_mapping_fn": policy_mapping_fn,
        },
        #"momentum": tune.uniform(0, 1),
        "num_workers": 0,  # parallelism
        "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
        "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        "clip_param": 0.5,
        "vf_clip_param": 5.0,
        "exploration_config": {
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 53,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            "feature_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
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


    stop = {
        "training_iteration": 100000,   # The number of times tune.report() has been called
        "timesteps_total": 10000000,   # Total number of timesteps
        #"episode_reward_mean": -0.1, # When to stop.. it would be great if we could define this in terms
                                    # of a more complex expression which incorporates the episode reward min too
                                    # There is a lot of variance in the episode reward min
    }

    #checkpoint = '/Users/mylesfoley/Desktop/Imperial/git/turing/cage-challenge-2/logs/PPO_B_lineAgent_2022-04-08_11-57-03/PPO_CybORGAgent_9f856_00000_0_2022-04-08_11-57-03/checkpoint_001522/checkpoint-1522'
    #local_dir_resume = log_dir + 'PPO_CUR_2022-02-24_MEANDER_3M/PPO_CybORGAgent_3ec94_00000_0_2022-02-24_18-04-44/'
    #agent = ppo.PPOTrainer(config=config, env=CybORGAgent)
    #agent.restore(checkpoint)
    log_dir = '../../logs/feudal'
    if len(sys.argv[1:]) != 1:
        print('No log directory specified, defaulting to: {}'.format(log_dir))
    else:
        log_dir = sys.argv[1]
        print('Log directory specified: {}'.format(log_dir))

    algo = ppo.PPOTrainer
    analysis = tune.run(algo, # Algo to use - alt: ppo.PPOTrainer, impala.ImpalaTrainer
                        config=config,
                        name=algo.__name__ + '_feudal_' + time.strftime("%Y-%m-%d_%H-%M-%S"),
                        local_dir=log_dir,
                        stop=stop,
                        #restore=checkpoint,
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        keep_checkpoints_num=3,
                        checkpoint_score_attr="episode_reward_min")

    checkpoint_pointer = open("checkpoint_pointer.txt", "w")
    last_checkpoint = analysis.get_last_checkpoint(
        metric="episode_reward_mean", mode="max"
    )

    checkpoint_pointer.write(last_checkpoint)
    print("Best model checkpoint written to: {}".format(last_checkpoint))

    # If you want to throw an error
    #if True:
    #    check_learning_achieved(analysis, 0.1)

    checkpoint_pointer.close()
    ray.shutdown()

    # You can run tensorboard --logdir=log_dir/PPO... to visualise the learning processs during and after training