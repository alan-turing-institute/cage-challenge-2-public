import numpy as np

from neural_nets import *
from hier_env import HierEnv
import os

from CybORG.Agents import B_lineAgent, SleepAgent, RedMeanderAgent
from configs import *
from CybORGActionAgent import CybORGActionAgent
class LoadHeuristicBlueAgent:

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
        #self.CTRL_checkpoint_pointer = two_up + '/logs/hier/PPO_hier_2022-07-11_10-57-15/PPO_HierEnv_d78d3_00000_0_2022-07-11_10-57-15/checkpoint_000347/checkpoint-347'
        self.BL_checkpoint_pointer = two_up + sub_agents['B_line_trained']
        self.RM_checkpoint_pointer = two_up + sub_agents['RedMeander_trained']

        #with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        #print("Using checkpoint file (Controller): {}".format(self.CTRL_checkpoint_pointer))
        print("Using checkpoint file (B-line): {}".format(self.BL_checkpoint_pointer))
        print("Using checkpoint file (Red Meander): {}".format(self.RM_checkpoint_pointer))

        config = PPO_Curiosity_config
        config['model']['fcnet_hiddens'] = [256, 256]
        config['env'] = HierEnv

        # Restore the controller model
        #self.controller_agent = ppo.PPOTrainer(config=config, env=HierEnv)
        #self.controller_agent.restore(self.CTRL_checkpoint_pointer)
        #self.observation = np.zeros((HierEnv.mem_len,52))

        RM_config = LSTM_config
        RM_config["in_evaluation"] = True
        RM_config["explore"] = False

        BL_config = PPO_Curiosity_config
        #BL_config['model']['fcnet_hiddens'] = [256, 256, 256]
        BL_config["in_evaluation"] = True
        BL_config["explore"] = False

        #load agent trained against RedMeanderAgent
        self.RM_def = ppo.PPOTrainer(config=RM_config, env=CybORGAgent)
        self.RM_def.restore(self.RM_checkpoint_pointer)
        #load agent trained against B_lineAgent
        BL_config['env'] = CybORGActionAgent
        BL_config["env_config"] = {'agent_name': 'Blue', 'env': None, 'max_steps': 100, 'attacker': B_lineAgent}
        self.BL_def = ppo.PPOTrainer(config=BL_config, env=CybORGActionAgent)
        self.BL_def.restore(self.BL_checkpoint_pointer)

        #self.red_agent=-1
        self.state = [np.zeros(256, np.float32),
                      np.zeros(256, np.float32)]
        self.step = 0
        # heuristics
        self.set = False
        self.observations = []
        self.adversary = 'B_line'


    def set_red_agent(self, red_agent):
        self.red_agent = red_agent

    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        #update sliding window
        # discover network services sequence
        self.observations.append(obs)
        self.step += 1
        if self.step == 4:
            if any(np.array_equal(host_observation, np.array([1,0,0,0]))for host_observation in self.observations[-1][2:].reshape(13, 4)) \
                and any(np.array_equal(host_observation, np.array([1,0,0,0]))for host_observation in self.observations[-2][2:].reshape(13, 4)) \
                    and not np.array_equal(self.observations[-1][2:], self.observations[-2][2:]):
                self.adversary = 'Meander'
            else:
                 self.adversary = 'B_line'


        #self.adversary = 'Meander'
        #print(self.adversary)

        #self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        #self.observation[HierEnv.mem_len-1] = obs           # Replace what's on the rightmost position

        #select agent to compute action
        """ if self.red_agent == B_lineAgent or self.red_agent == SleepAgent:
            agent_to_select = 0
        else: #RedMeanderAgent
            agent_to_select = 1"""
        #agent_to_select = self.controller_agent.compute_single_action(obs)
        if self.adversary == 'B_line':
            #print('b_line defence')
            # get action from agent trained against the B_lineAgent

            # keep track of the lstm state for later use
            _, self.state, _ = self.RM_def.compute_single_action(obs[2:], self.state)
            agent_action = self.BL_def.compute_single_action(obs)
        elif self.adversary == 'Meander':
            # get action from agent trained against the RedMeanderAgent
            agent_action, state, _ = self.RM_def.compute_single_action(obs[2:], self.state)
            #print('meander defence')
            self.state = state
            # agent_action = self.RM_def.compute_single_action(self.observation[-1:])
        else:
            print('something went terribly wrong, old sport')
        return agent_action, self.adversary

    def end_episode(self):
        self.set = False
        self.state = [np.zeros(256, np.float32),
               np.zeros(256, np.float32)]
        self.observations = []
        self.step = 0