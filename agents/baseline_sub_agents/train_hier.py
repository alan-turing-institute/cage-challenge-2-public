"""Alternative RLLib model based on local training

You can visualize experiment results in ~/ray_results using TensorBoard.
"""

import sys
# Ray imports
import ray
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from hier_env import HierEnv
import time
from configs import *
from neural_nets import *
tf1, tf, tfv = try_import_tf()

if __name__ == "__main__":
    ray.init()

    # Can also register the env creator function explicitly with register_env("env name", lambda config: EnvClass(config))
    ModelCatalog.register_custom_model("CybORG_hier_Model", TorchModel)
    config = PPO_Curiosity_config
    config['model']['fcnet_hiddens'] = [256, 256]
    config['env'] = HierEnv

    stop = {
        "training_iteration": 100000,   # The number of times tune.report() has been called
        "timesteps_total": 10000000,   # Total number of timesteps
        "episode_reward_mean": -0.1, # When to stop.. it would be great if we could define this in terms
                                    # of a more complex expression which incorporates the episode reward min too
                                    # There is a lot of variance in the episode reward min
    }

    log_dir = '../../logs/hier'
    if len(sys.argv[1:]) != 1:
        print('No log directory specified, defaulting to: {}'.format(log_dir))
    else:
        log_dir = sys.argv[1]
        print('Log directory specified: {}'.format(log_dir))

    algo = ppo.PPOTrainer
    analysis = tune.run(ppo.PPOTrainer, # Algo to use - alt: ppo.PPOTrainer, impala.ImpalaTrainer
                        config=config,
                        name=algo.__name__ + '_' +'hier'+ '_' + time.strftime("%Y-%m-%d_%H-%M-%S"),
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

    # If you want to throw an error
    #if True:
    #    check_learning_achieved(analysis, 0.1)

    checkpoint_pointer.close()
    ray.shutdown()

    # You can run tensorboard --logdir=log_dir/PPO... to visualise the learning processs during and after training

