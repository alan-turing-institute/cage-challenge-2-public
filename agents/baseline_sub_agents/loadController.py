
from neural_nets import *
from hier_env import HierEnv
import os
from CybORG.Agents import B_lineAgent, SleepAgent, RedMeanderAgent
from configs import *
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
        self.CTRL_checkpoint_pointer = two_up + '/logs/hier/PPO_hier_2022-07-11_10-57-15/PPO_HierEnv_d78d3_00000_0_2022-07-11_10-57-15/checkpoint_000255/checkpoint-255'
        self.BL_checkpoint_pointer = two_up + sub_agents['B_line_trained']
        self.RM_checkpoint_pointer = two_up + sub_agents['RedMeander_trained']

        #with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        print("Using checkpoint file (Controller): {}".format(self.CTRL_checkpoint_pointer))
        print("Using checkpoint file (B-line): {}".format(self.BL_checkpoint_pointer))
        print("Using checkpoint file (Red Meander): {}".format(self.RM_checkpoint_pointer))

        config = PPO_Curiosity_config
        config['model']['fcnet_hiddens'] = [256, 256]
        config['env'] = HierEnv

        # Restore the controller model
        self.controller_agent = ppo.PPOTrainer(config=config, env=HierEnv)
        self.controller_agent.restore(self.CTRL_checkpoint_pointer)
        self.observation = np.zeros((HierEnv.mem_len,52))

        RM_config = LSTM_config
        RM_config["in_evaluation"] = True
        RM_config["explore"] = False

        BL_config = PPO_Curiosity_config
        BL_config['model']['fcnet_hiddens'] = [256, 256, 256]
        BL_config["in_evaluation"] = True
        BL_config["explore"] = False

        #load agent trained against RedMeanderAgent
        self.RM_def = ppo.PPOTrainer(config=RM_config, env=CybORGAgent)
        self.RM_def.restore(self.RM_checkpoint_pointer)
        #load agent trained against B_lineAgent
        self.BL_def = ppo.PPOTrainer(config=BL_config, env=CybORGAgent)
        self.BL_def.restore(self.BL_checkpoint_pointer)

        #self.red_agent=-1
        self.state = [np.zeros(256, np.float32),
                      np.zeros(256, np.float32)]


    def set_red_agent(self, red_agent):
        self.red_agent = red_agent

    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        #update sliding window
        self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        self.observation[HierEnv.mem_len-1] = obs           # Replace what's on the rightmost position

        #select agent to compute action
        """ if self.red_agent == B_lineAgent or self.red_agent == SleepAgent:
            agent_to_select = 0
        else: #RedMeanderAgent
            agent_to_select = 1"""
        agent_to_select = self.controller_agent.compute_single_action(self.observation)

        if agent_to_select == 0:
            # get action from agent trained against the B_lineAgent
            agent_action = self.BL_def.compute_single_action(self.observation[-1:])

            # keep track of the lstm state for later use
            _, self.state, _ = self.RM_def.compute_single_action(self.observation[-1:], self.state)
        elif agent_to_select == 1:
            # get action from agent trained against the RedMeanderAgent
            agent_action, self.state, _ = self.RM_def.compute_single_action(self.observation[-1:], self.state)
            # self.state = state
            # agent_action = self.RM_def.compute_single_action(self.observation[-1:])
        else:
            print('something went terribly wrong, old sport')
        return agent_action, agent_to_select

    def end_episode(self):
        self.state = [np.zeros(256, np.float32),
               np.zeros(256, np.float32)]