import os
from pprint import pprint
import os.path as path
import numpy as np
import ray
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from train_hier import CustomModel, TorchModel
from CybORGHier import HierEnv
import os
from CybORG.Agents import B_lineAgent, SleepAgent, RedMeanderAgent
from sub_agents import sub_agents
from CybORGAgent import CybORGAgent
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG

class LoadBlueAgent:

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
        self.CTRL_checkpoint_pointer = two_up + '/logs/PPO_Hier_20220410_184257/PPO_HierEnv_a83db_00000_0_2022-04-10_18-42-57/checkpoint_001316/checkpoint-1316'
        self.BL_checkpoint_pointer = two_up + sub_agents['B_line_trained']
        self.RM_checkpoint_pointer = two_up + sub_agents['RedMeander_trained']

        #with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        print("Using checkpoint file (Controller): {}".format(self.CTRL_checkpoint_pointer))
        print("Using checkpoint file (B-line): {}".format(self.BL_checkpoint_pointer))
        print("Using checkpoint file (Red Meander): {}".format(self.RM_checkpoint_pointer))

        config = Trainer.merge_trainer_configs(
            DEFAULT_CONFIG,
            {
            "env": HierEnv,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env various set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_hier_Model",
                "vf_share_layers": True,
            },
            "lr": 0.0001,
            #"momentum": tune.uniform(0, 1),
            "num_workers": 4,  # parallelism
            "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
             "in_evaluation": True,
            'explore': False
        })

        # Restore the controller model
        self.controller_agent = ppo.PPOTrainer(config=config, env=HierEnv)
        self.controller_agent.restore(self.CTRL_checkpoint_pointer)
        self.observation = np.zeros((HierEnv.mem_len,52))

        subagent_config = Trainer.merge_trainer_configs(
            DEFAULT_CONFIG, {
                "env": CybORGAgent,
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
                # "momentum": tune.uniform(0, 1),
                "num_workers": 0,  # parallelism
                "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
                "eager_tracing": True,
                # In order to reach similar execution speed as with static-graph mode (tf default)
                "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
                "clip_param": 0.5,
                "vf_clip_param": 5.0,
                "in_evaluation": True,
                'explore': False,
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

        #load agent trained against RedMeanderAgent
        self.RM_def = ppo.PPOTrainer(config=subagent_config, env=CybORGAgent)
        self.RM_def.restore(self.RM_checkpoint_pointer)
        #load agent trained against B_lineAgent
        self.BL_def = ppo.PPOTrainer(config=subagent_config, env=CybORGAgent)
        self.BL_def.restore(self.BL_checkpoint_pointer)

        self.red_agent=-1


    def set_red_agent(self, red_agent):
        self.red_agent = red_agent

    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        #update sliding window
        self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        self.observation[HierEnv.mem_len-1] = obs           # Replace what's on the rightmost position

        #select agent to compute action
        if self.red_agent == B_lineAgent or self.red_agent == SleepAgent:
            agent_to_select = 0
        else: #RedMeanderAgent
            agent_to_select = 1

        #self.controller_agent.compute_single_action(self.observation)
        #agent_to_select = 1#np.random.choice([0,1]) # hard-coded meander agent only
        if agent_to_select == 0:
            # get action from agent trained against the B_lineAgent
            agent_action = self.BL_def.compute_single_action(self.observation[-1:])
        elif agent_to_select == 1:
            # get action from agent trained against the RedMeanderAgent
            agent_action = self.RM_def.compute_single_action(self.observation[-1:])
        return agent_action#, agent_to_select