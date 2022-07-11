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
from gym.spaces import Box, Tuple

from configs import *
from rnn import TorchRNNModel
from CybORGActionAgent import CybORGActionAgent
tf1, tf, tfv = try_import_tf()
from ray.rllib.agents.callbacks import DefaultCallbacks


@ray.remote
class LevelManager:
    step      = 1
    horizon   = 700

    def __init__(self):
        self.level     = 0
        self.max_level = 100
        self.rewards   = []

    def flush(self):
        self.level     = 0
        self.max_level = 100
        self.rewards   = []

    def set_range(self, start, end):
        self.level = start
        self.max_level = end

    def get_level(self):
        return self.level

    def append_reward(self, lvl, reward):
        self.rewards[lvl].append(reward)

    def append_reward_batch(self, lvl, reward_batch):
        if lvl == self.level: # Ignore any rewards from past levels
            self.rewards.extend(reward_batch)
            print("Appended", len(reward_batch), "samples for level", lvl,  ".", "The total number of samples is:", len(self.rewards), flush=True)
        else:
            print("Stale rewards submitted. Ignoring. ( Current lvl:", self.level, " |  Submitted lvl:", lvl, ").", flush=True)

    def levelup(self):
        if self.level >= self.max_level:
            return self.level

        samples = self.rewards[-self.horizon:]
        if len(samples) < self.horizon:
            return self.level

        avg_score = sum(samples)/len(samples)


        if (avg_score > -1):
            self.level += self.step
            print(">>> Leveled up to", self.level, "steps. With score:", avg_score, flush=True)
            self.reset()


        #print(">>>>>", self.level, flush=True)
        return self.level

    def reset(self):
        self.rewards = []

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    #agent_idx = int(agent_id[-1])  # 0 (player1) or 1 (player2)
    # agent_id = "player[1|2]" -> policy depends on episode ID
    # This way, we make sure that both policies sometimes play player1
    # (start player) and sometimes player2 (player to move 2nd).
    return "standard_policy" if episode.succes == 'TRUE' else "fixer_policy"



if __name__ == "__main__":
    ray.init()




    # Can also register the env creator function explicitly with register_env("env name", lambda config: EnvClass(config))
    ModelCatalog.register_custom_model("CybORG_PPO_Model", GTrXLNet)
    ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)
    ModelCatalog.register_custom_model("CybORG_A2C_Model", TorchModel)
    ModelCatalog.register_custom_model("CybORG_RNN_Model", TorchRNNModel)

    #dummy_env = CybORGActionAgent()
    #from CybORGMultiAdversaryAgent import CybORGMultiAgent
    config = PPO_Curiosity_config
    config['multiagent'] = {'policies':
                                {"standard_policy":( None,
                                        CybORGActionAgent.observation_space,
                                        CybORGActionAgent.action_space,
                                        {"gamma": 0.0}),
                                 "fixer_policy": ( None,
                                        CybORGActionAgent.observation_space,
                                        CybORGActionAgent.action_space,
                                        {"gamma": 0.0}),
                                 "policies_to_train": lambda pid, batch: len(batch.agent_steps()) == 0
                                 }
                           }
    config['num_workers'] = 0
    config['env'] = CybORGActionAgent

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
                        name=algo.__name__ + '_' +CybORGAgent.agents['Red'].__name__+ '_' + time.strftime("%Y-%m-%d_%H-%M-%S"),
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