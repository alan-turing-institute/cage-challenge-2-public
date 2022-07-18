import gym, inspect, random
from ray.rllib.env.env_context import EnvContext

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, EnumActionWrapper
from BlueTableActionWrapper import  BlueTableWrapper
import numpy as np
from gym import Env, spaces

class CybORGActionAgent(Env, BaseWrapper):
    max_steps = 100
    #path = str(inspect.getfile(CybORG))
    #path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    #agents = {
    #    'Red': B_lineAgent  # , #RedMeanderAgent, 'Green': GreenAgent
    #}
    def __init__(self, config):
        super().__init__()
        self.agent_name = config['agent_name']
        if config['env'] is not None:
            self.cyborg = config['env']
            self.agents = {
                'Red': config['attacker']  # , #RedMeanderAgent, 'Green': GreenAgent
            }
        else:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
            self.agents = {
                'Red': config['attacker']#B_lineAgent  # , #RedMeanderAgent, 'Green': GreenAgent
            }
            self.cyborg = CybORG(path, 'sim', agents=self.agents)

        #self.agent_name = self.agent_name
        self.env = BlueTableWrapper(self.cyborg, output_mode='vector')
        self.env = EnumActionWrapper(self.env)
        self.env = OpenAIGymWrapper(agent_name=self.agent_name, env=self.env)

        #self.env = self.cyborg
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(54,), dtype=np.float32)
        #self.reward_threshold = reward_threshold
        self.max_steps = config['max_steps']
        self.step_counter = 0
        #self.agent_name = self.env.agent_name
        self.action = None
        self.success = None

    def step(self, action=None):
        obs, reward, done, info = self.env.step(action=action)
        assert (obs.shape[0] == 54)
        if np.array_equal(obs[:2], [0,1]):
            self.success = False
        elif np.array_equal(obs[:2], [1,0]):
            self.success = True
        else:
            self.success = False
        #obs = obs[2:]
        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True
        assert (self.step_counter <= self.max_steps)

        return obs, reward, done, info

    def reset(self):
        self.step_counter = 0
        return self.env.reset()

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self, agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr('get_reward_breakdown')(agent)