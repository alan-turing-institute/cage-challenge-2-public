import gym, inspect, random
from ray.rllib.env.env_context import EnvContext

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
from CybORG.Agents.Wrappers import ChallengeWrapper, OpenAIGymWrapper, EnumActionWrapper, FixedFlatWrapper, ReduceActionSpaceWrapper


class CybORGMultiAgent(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'


    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):


        self.cyborg = CybORG(self.path, 'sim', agents={'Red':RedMeanderAgent})
        self.RMenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.cyborg = CybORG(self.path, 'sim', agents={'Red':B_lineAgent})
        self.BLenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.env = self.BLenv
        self.steps = 1
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action = None


    def reset(self):
        self.steps = 1
        if random.choice([0,1]) == 0:
            self.env = self.BLenv
        else:
            self.env = self.RMenv
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

