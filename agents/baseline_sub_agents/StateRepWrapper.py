from gym import Env, spaces
from gym.spaces import Discrete, Tuple
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper,RedTableWrapper,EnumActionWrapper, FixedFlatWrapper
from CybORG.Agents.SimpleAgents import BaseAgent
from CybORG.Shared import Results
from typing import Union, List
import inspect
from prettytable import PrettyTable
import numpy as np
from pprint import pprint
import random

# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from newBlueTableWrapper import NewBlueTableWrapper
from newFlatWrapper import NewFixedFlatWrapper

# import decoy actions
from CybORG.Shared.Actions.ConcreteActions.DecoyApache import DecoyApache
from CybORG.Shared.Actions.ConcreteActions.DecoyFemitter import DecoyFemitter
from CybORG.Shared.Actions.ConcreteActions.DecoyHarakaSMPT import DecoyHarakaSMPT
from CybORG.Shared.Actions.ConcreteActions.DecoySmss import DecoySmss
from CybORG.Shared.Actions.ConcreteActions.DecoySSHD import DecoySSHD
from CybORG.Shared.Actions.ConcreteActions.DecoySvchost import DecoySvchost
from CybORG.Shared.Actions.ConcreteActions.DecoyTomcat import DecoyTomcat
from CybORG.Shared.Actions.ConcreteActions.DecoyVsftpd import DecoyVsftpd

class NewEnumActionWrapper(BaseWrapper):
    def __init__(self, env: Union[type, BaseWrapper] = None, agent: BaseAgent = None, reduce_misinform=False):
        super().__init__(env, agent)
        self.possible_actions = None
        self.reduce_misinform = reduce_misinform
        self.action_signature = {}
        self.decoy_top = DecoyFemitter
        self.decoy_other = [DecoyApache, DecoyHarakaSMPT, DecoySmss, DecoySSHD, DecoySvchost, DecoyTomcat, DecoyVsftpd]
        self.decoy_top_idx = None
        self.decoy_other_idx = []
        self.get_action_space('Red')
        
    def step(self, agent=None, action: list = None) -> Results:
        if action is not None:
            if self.reduce_misinform:
                if action[0] == self.decoy_other_idx[0]:
                    a1 = random.choice(self.decoy_other_idx) 
                elif action[0] == 6:
                    a1 = 12 # map to the last type of action in self.possible_actions
                else: 
                    a1 = action[0]
            else:
                a1 = action[0]
            if len(self.possible_actions[a1])==1:
                action = self.possible_actions[a1]
            else:
                action = self.possible_actions[a1][action[1]] ## changed here
        return super().step(agent, action)

    def action_space_change(self, action_space: dict) -> int:
        assert type(action_space) is dict, \
            f"Wrapper required a dictionary action space. " \
            f"Please check that the wrappers below the ReduceActionSpaceWrapper return the action space as a dict "
        possible_actions = []
        self.decoy_other_idx = []
        temp = {}
        params = ['action']
        for i, action in enumerate(action_space['action']):
            # check the decoy index:
            if action is self.decoy_top:
                self.decoy_top_idx  = i
            elif action in self.decoy_other: 
                self.decoy_other_idx += [i]
            
            # find all possible actions for each type of action
            if action not in self.action_signature:
                self.action_signature[action] = inspect.signature(action).parameters
            param_dict = {}
            param_list = [{}]
            for p in self.action_signature[action]:
                if p == 'priority':
                    continue
                temp[p] = []
                if p not in params:
                    params.append(p)

                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            p_dict[p] = key
                            new_param_list.append({key: value for key, value in p_dict.items()})
                    param_list = new_param_list

        # create nested list for each type of action
            possible_actions_temp = []
            for p_dict in param_list:
                possible_actions_temp.append(action(**p_dict))
            possible_actions += [possible_actions_temp]

        self.possible_actions = possible_actions
        # print('\033[92m' + 'possible_actions:' + '\033[0m', flush=True)
        # print(np.shape(possible_actions))

        # find the action space
        if self.reduce_misinform:
            action1_len = len(action_space['action']) - len(self.decoy_other_idx) + 1
        else: 
            action1_len = len(action_space['action'])
        action2_len = len(max(possible_actions, key=lambda action: len(action)))
        if action_space['agent']  != {'Red': True}:
            assert action2_len == len(action_space['hostname']), f"second action has length of {action2_len}, "\
                                                                f"while there are {len(action_space['hostname'])} number of hosts"

        return [action1_len, action2_len]



class NewOpenAIGymWrapper(Env, BaseWrapper):
    def __init__(self, agent_name: str, env: BaseWrapper = None, agent: BaseAgent = None):
        super().__init__(env, agent_name)
        self.agent_name = agent_name
        if isinstance(self.get_action_space(self.agent_name), list):
            # self.action_space = spaces.MultiDiscrete(self.get_action_space(self.agent_name))
            action_space_shape = self.get_action_space(self.agent_name)
            self.action_space = Tuple([Discrete(action_space_shape[0]), Discrete(action_space_shape[1])])
        else:
            assert isinstance(self.get_action_space(self.agent_name), int)
            self.action_space = spaces.Discrete(self.get_action_space(self.agent_name))
        box_len = len(self.observation_change(self.env.reset(self.agent_name).observation))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(box_len,), dtype=np.float32)
        self.reward_range = (float('-inf'), float('inf'))
        self.metadata = {}
        self.action = None

    def step(self, action: Union[int, List[int]] = None): 
        self.action = action
        # print('OpenAI action', action, flush=True)

        result = self.env.step(self.agent_name, action)
        result.observation = self.observation_change(result.observation)
        result.action_space = self.action_space_change(result.action_space)
        info = vars(result)
        return np.array(result.observation), result.reward, result.done, info

    def reset(self, agent=None):
        result = self.env.reset(self.agent_name)
        result.action_space = self.action_space_change(result.action_space)
        result.observation = self.observation_change(result.observation)
        return np.array(result.observation)

    def render(self):
        if self.agent_name == 'Red':
            table = PrettyTable({
                'Subnet',
                'IP Address',
                'Hostname',
                'Scanned',
                'Access',
            })
            for ip in self.get_attr('red_info'):
                table.add_row(self.get_attr('red_info')[ip])
            table.sortby = 'IP Address'
            if self.action is not None:
                _action = self.get_attr('possible_actions')[self.action[0]][self.action[1]] # change action selection
                return print(f'\nRed Action: {_action}\n{table}')
        elif self.agent_name == 'Blue':
            table = PrettyTable({
                'Subnet',
                'IP Address',
                'Hostname',
                'Activity',
                'Compromised',
            })
            for hostid in self.get_attr('info'):
                table.add_row(self.get_attr('info')[hostid])
            table.sortby = 'Hostname'
            if self.action is not None:
                _action = self.get_attr('possible_actions')[self.action[0]][self.action[1]] # change action selection
                red_action = self.get_last_action(agent=self.agent_name)
                return print(f'\nBlue Action: {_action}\nRed Action: {red_action}\n{table}')
        return print(table)

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self,agent:str):
        return self.get_attr('get_agent_state')(agent)

    def get_action_space(self,agent):
        return self.env.get_action_space(agent)

    def get_last_action(self,agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()



class StateRepWrapper(Env,BaseWrapper):
    def __init__(self, agent_name: str, actionSpace: int, obsSpace: int, env, agent=None,
            reward_threshold=None, max_steps = None):
        """
        actionSpace: 
            0 -> list of 145 possible actions; 
            1 -> tuple(13, 13) autoregressive;
            2 -> tuple(7, 13) reduce misinform action;
        obsSpace: 
            0 -> table wrapper observation (challenge wrapper);
            1 -> fixed flat wrapper observation;
            2 -> new FlatWrapper
            3 -> new BlueTableWrapper
        """
        super().__init__(env, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')

        # observation space
        if obsSpace ==0:
            env = table_wrapper(env, output_mode='vector')
        elif obsSpace == 1:
            env = FixedFlatWrapper(env)
        elif obsSpace == 2:
            env = NewFixedFlatWrapper(env=env)
        elif obsSpace == 3:
            env = NewBlueTableWrapper(env=env, output_mode='float_vector', port_length=0)
        else: 
            raise ValueError('Please input valid integer for obsSpace argument in StateRepWrapper()')
            
        # action space
        if actionSpace == 0:
            env = EnumActionWrapper(env=env)
        elif actionSpace == 1:
            env = NewEnumActionWrapper(env, reduce_misinform=False)
        elif actionSpace == 2:
            env = NewEnumActionWrapper(env, reduce_misinform=True)

        env = NewOpenAIGymWrapper(agent_name=agent_name, env=env)

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None

    def step(self,action=None):
        obs, reward, done, info = self.env.step(action=action)
        
        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True

        return obs, reward, done, info

    def reset(self):
        self.step_counter = 0
        return self.env.reset()

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self,agent:str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self,agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr('get_reward_breakdown')(agent)

