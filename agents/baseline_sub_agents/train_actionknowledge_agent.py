"""Alternative RLLib model based on local training
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import sys

# Ray imports
import ray
from ray import tune
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a2c
import ray.rllib.agents.sac as sac
import ray.rllib.agents.maml as maml
import ray.rllib.agents.impala as impala

from ray.rllib.utils.framework import try_import_tf, try_import_torch
import time
from configs import *
from rnn import TorchRNNModel
from CybORGActionAgent import CybORGActionAgent
tf1, tf, tfv = try_import_tf()



if __name__ == "__main__":
    ray.init()




    # Can also register the env creator function explicitly with register_env("env name", lambda config: EnvClass(config))
    ModelCatalog.register_custom_model("CybORG_PPO_Model", GTrXLNet)
    ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)
    ModelCatalog.register_custom_model("CybORG_A2C_Model", TorchModel)
    ModelCatalog.register_custom_model("CybORG_RNN_Model", TorchRNNModel)

    """config = Trainer.merge_trainer_configs(
        PPO_CONFIG, {
            "env": CybORGAgent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
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
        })"""

    #from CybORGMultiAdversaryAgent import CybORGMultiAgent
    config = PPO_Curiosity_config
    config['num_workers'] = 0
    config['env'] = CybORGActionAgent
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

    #cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
    config["env_config"] = {'agent_name': 'Blue', 'env':None, 'max_steps': 100}

    #from CybORGMultiAdversaryAgent import CybORGAgent
    #config['env'] = CybORGAgent
    #config['model']['custom_model'] = 'CybORG_RNN_Model'
    #config['model']['use_lstm'] = False
    stop = {
        "training_iteration": 100000,   # The number of times tune.report() has been called
        "timesteps_total": 5000000,   # Total number of timesteps
        "episode_reward_mean": -0.1, # When to stop.. it would be great if we could define this in terms
                                    # of a more complex expression which incorporates the episode reward min too
                                    # There is a lot of variance in the episode reward min
    }

    checkpoint = '/Users/mylesfoley/Desktop/Imperial/git/turing/cage-challenge-2/logs/various/PPO_curiosity_2_layer_2022-07-08_17-25-48/PPO_CybORGAgent_a039e_00000_0_2022-07-08_17-25-48/checkpoint_001250/checkpoint-1250'
    #local_dir_resume = log_dir + 'PPO_CUR_2022-02-24_MEANDER_3M/PPO_CybORGAgent_3ec94_00000_0_2022-02-24_18-04-44/'
    #agent = ppo.PPOTrainer(config=config, env=CybORGAgent)
    #agent.restore(checkpoint)
    log_dir = '../../logs/various'
    if len(sys.argv[1:]) != 1:
        print('No log directory specified, defaulting to: {}'.format(log_dir))
    else:
        log_dir = sys.argv[1]
        print('Log directory specified: {}'.format(log_dir))
    #check its not the maml env
    algo = ppo.PPOTrainer
    analysis = tune.run(algo, # Algo to use - alt: ppo.PPOTrainer, impala.ImpalaTrainer
                        config=config,
                        name=algo.__name__ + '_action_knowledge_' +CybORGAgent.agents['Red'].__name__+ '_' + time.strftime("%Y-%m-%d_%H-%M-%S"),
                        #name=algo.__name__ + '_curiosity_2_layer_' + time.strftime("%Y-%m-%d_%H-%M-%S"),
                        local_dir=log_dir,
                        stop=stop,
                        #restore=checkpoint,
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        keep_checkpoints_num=3,
                        checkpoint_score_attr="episode_reward_mean")

    #checkpoint_pointer = open("../hier_extended/checkpoint_pointer.txt", "w")
    #last_checkpoint = analysis.get_last_checkpoint(
    #    metric="episode_reward_mean", mode="max"
    #)

    #checkpoint_pointer.write(last_checkpoint)
    #print("Best model checkpoint written to: {}".format(last_checkpoint))

    # If you want to throw an error
    #if True:
    #    check_learning_achieved(analysis, 0.1)

    #checkpoint_pointer.close()
    ray.shutdown()

    # You can run tensorboard --logdir=log_dir/PPO... to visualise the learning processs during and after training