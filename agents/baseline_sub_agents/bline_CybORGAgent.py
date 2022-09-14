from cv2 import reduce
import gym, inspect, random
from ray.rllib.env.env_context import EnvContext
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
# from CybORG.Agents.Wrappers import ChallengeWrapper
from StateRepWrapper import StateRepWrapper
from pprint import pprint
# from csv import writer

class CybORGAgent(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

    agents = {
        'Red':  B_lineAgent
    }

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents=self.agents)

        self.env  = StateRepWrapper(env=self.cyborg, agent_name='Blue', actionSpace=0, obsSpace=3)
        self.steps = 0
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space
        print('\033[92m' + '///// initiating CybORG... action space:' + '\033[0m', flush=True)
        pprint(self.action_space)
        self.observation_space = self.env.observation_space
        self.action = None

    def reset(self):
        self.steps = 1
        return self.env.reset()
    
    def step(self, action=None):

        result = self.env.step(action=action)

        self.steps += 1
        if self.steps == self.max_steps:
            return result[0], result[1], True, result[3]
        assert (self.steps <= self.max_steps)
        return result

    def seed(self, seed=None):
        random.seed(seed)

