"""Alternative RLLib model based on local training
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
##############################
import os
import sys
from neural_nets import *
from sub_agents import *
from configs import *

# Ray imports
import ray
from ray import tune
from ray.tune import grid_search
from ray.tune.schedulers import ASHAScheduler # https://openreview.net/forum?id=S1Y7OOlRZ algo for early stopping
from ray.rllib.agents.trainer import Trainer
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
import time
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


from typing import Any
from pprint import pprint
#import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2

from CybORGAgent import CybORGAgent




#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
# torch.device(str("cuda:0"))


if __name__ == "__main__":
    # set subagent config
    config = dict()
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'bline':
            config = bline_config
        elif sys.argv[1].lower() == 'meander':
            config = meander_config
        else:
            raise ValueError("Please specify which subagent you'd like to train: \"bline\" or \"meander\".")
    else: 
        raise ValueError("Please specify which subagent you'd like to train: \"bline\" or \"meander\".")
    
    adversary_name = config['env'].agents['Red'].__name__
    # ModelCatalog.register_custom_model("CybORG_hier_Model", TorchModel)
    print('\033[92m' + "/"*50 + '\033[0m', flush=True)
    print('\033[92m' + "Training defender for " + adversary_name + "..." + '\033[0m', flush=True)
    ray.init()    
    
    # gpu availability
    print("torch.cuda.is_available()", torch.cuda.is_available())
    torch.device(str("cuda:0"))
    if torch.cuda.is_available():
        torch.device(str("cuda:0"))
        gpus = 1
    else: 
        gpus = 0
    config['num_gpus'] = gpus


    stop = {
        "training_iteration": 10000000,   # The number of times tune.report() has been called
        "timesteps_total": 10000000,   # Total number of timesteps
        "episode_reward_mean": -0.1, # When to stop.. it would be great if we could define this in terms
                                    # of a more complex expression which incorporates the episode reward min too
                                    # There is a lot of variance in the episode reward min
    }

    log_dir = '../../logs/training/'
    algo = ppo.PPOTrainer
    analysis = tune.run(algo,
                        config=bline_config,
                        name= algo.__name__ + '_' + adversary_name + '_' + time.strftime("%Y-%m-%d_%H-%M-%S"),
                        local_dir=log_dir,
                        stop=stop,
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        keep_checkpoints_num=3,
                        checkpoint_score_attr="episode_reward_mean")

    checkpoint_pointer = open("checkpoint_pointer.txt", "w")
    last_checkpoint = analysis.get_last_checkpoint(
        metric="episode_reward_mean", mode="max"
    )

    checkpoint_pointer.write(last_checkpoint)
    print("Best model checkpoint written to: {}".format(last_checkpoint))
    
    print(algo.get_weights())
    checkpoint_pointer.close()
    ray.shutdown()

    # You can run tensorboard --logdir=logs to visualise the learning processs during and after training
    # tensorboard --logdir=thesis_logs