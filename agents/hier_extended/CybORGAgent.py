import gym, inspect, random
from ray.rllib.env.env_context import EnvContext

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
from CybORG.Agents.Wrappers import ChallengeWrapper


class CybORGAgent(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

    agents = {
        'Red': RedMeanderAgent
        #B_lineAgent  # , #RedMeanderAgent, 'Green': GreenAgent
    }

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents=self.agents)

        #self.env = OpenAIGymWrapper('Blue',
        #                            EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(self.cyborg))))
        self.env  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        #self.action_space = 54
        self.steps = 0
        self.agent_name = self.env.agent_name
        if isinstance(self.env.get_action_space(self.agent_name), list):
            self.action_space = gym.spaces.MultiDiscrete(54)
        else:
            assert isinstance(self.env.get_action_space(self.agent_name), int)
            self.action_space = gym.spaces.Discrete(54)
        self.observation_space = self.env.observation_space
        self.action = None

    def reset(self):
        self.steps = 1
        return self.env.reset()

    def step(self, action=None):
        if action > 40:
            action += 104
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == self.max_steps:
            return result[0], result[1], True, result[3]
        assert (self.steps <= self.max_steps)
        return result

    def seed(self, seed=None):
        random.seed(seed)

