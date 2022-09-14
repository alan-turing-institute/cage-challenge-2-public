from CybORG import CybORG
from CybORG.Agents import *
from CybORG.Agents.Wrappers import ChallengeWrapper
import inspect
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import random
import pickle as pkl
import time

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
agents = {
    'Red': B_lineAgent  # RedMeanderAgent
}

env = CybORG(path, 'sim', agents={'Red': B_lineAgent})
BL_env = ChallengeWrapper(env=env, agent_name='Blue')  # wrap(cyborg)

env = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
RM_env = ChallengeWrapper(env=env, agent_name='Blue')  # wrap(cyborg)

env = CybORG(path, 'sim', agents={'Red': SleepAgent})
SL_env = ChallengeWrapper(env=env, agent_name='Blue')  # wrap(cyborg)

action_map = {
    0: B_lineAgent,
    1: RedMeanderAgent,
    2: SleepAgent,
}
action_selector = list(range(len(action_map)))

red_states = []
with open('red_states.pkl', "rb") as bline_states_file:  # Must open file in binary mode for pickle
    print('Red Agent states loaded from {}'.format(bline_states_file))
    red_states = pkl.load(bline_states_file)

N_EPISODES = 15000
N_STEPS = 4
EPSILON = 0.01
test_episodes = 1000
def init_bandit(epsilon=0.0001):
    # Generate an arbitrary value function and (approximately..) soft policy
    Q = {}
    N = {}
    rewards = {}
    #policy = {}
    #returns = list()
    # Policy is a map from obs to action
    for s in red_states:
        s_hashable = ''.join(str(bit) for bit in s)
        Q[s_hashable] = np.zeros(len(action_map))
        N[s_hashable] = {action:0 for action in action_map.keys()}
        rewards[s_hashable] = []
    #    #s_hashable = s  # ''.join(str(bit) for bit in s)
    #    #policy[s_hashable] = np.random.dirichlet(np.ones(n_blue_actions) * 100., size=1)[0]

    #Q = np.zeros(len(action_selector))#{action:0 for action in action_map.keys()}
    return Q, N, rewards#, policy, returns



Q, N, rewards = init_bandit(EPSILON)

import pandas as pd

dataset = pd.DataFrame(['observation', 'label'])

for episode in range(N_EPISODES):
    bandit_obs = np.array([], dtype=int)
    attacker =  np.random.choice([0, 1, 2], p=[0.475, 0.475, 0.05])
    if attacker == 0:
        current_env = BL_env
        adversary = B_lineAgent
    elif attacker == 1:
        current_env = RM_env
        adversary = RedMeanderAgent
    elif attacker == 2:
        current_env = SL_env
        adversary = SleepAgent


    blue_obs = current_env.reset()

    for step in range(N_STEPS):
        bandit_obs = np.append(bandit_obs, blue_obs)
        blue_obs, rew, done, info = current_env.step(0)
    bandit_obs = ''.join(str(bit) for bit in bandit_obs)

    dataset.append({'observation':bandit_obs, 'label':attacker}, ignore_index=True)

    if random.uniform(0, 1) < EPSILON:
        action = random.choice(action_selector)
    else:
        try:
            action = np.argmax(Q[bandit_obs])
        except KeyError as e:
            # State not seen before
            print('State not seen before: {}'.format(bandit_obs))
            # Initialise with a random soft policy
            Q[bandit_obs] = np.zeros(len(action_map))
            N[bandit_obs] = {action:0 for action in action_map.keys()}
            rewards[bandit_obs] = []
            action = np.argmax(Q[bandit_obs])

    if action == 0:
        #print('guess')
        if adversary == action_map[0]:
            bandit_rew = 1
        else:
            bandit_rew = -1
    elif action == 1:
        #print('guess')
        if adversary == action_map[1]:
            bandit_rew = 1
        else:
            bandit_rew = -1
    elif action == 2:
        bandit_rew = 0
    #increment actions seen
    N[bandit_obs][action] += 1
    # compute Q
    Q[bandit_obs][action] = Q[bandit_obs][action] + (1/(N[bandit_obs][action]))*(bandit_rew - Q[bandit_obs][action])

    rewards[bandit_obs].append(bandit_rew)

cumulative_average = rewards
for obs in rewards.keys():
    cumulative_average[obs] = np.cumsum(rewards[obs]) / len(rewards)
    #print(rewards[obs])

bandit_path = '../../logs/training/controller_bandit_{}'.format(time.strftime("%Y-%m-%d_%H-%M-%S"))
os.mkdir(bandit_path)
# plot moving average ctr
obs_seen = []
for obs in cumulative_average.keys():
    if sum(cumulative_average[obs]) != 0:
        plt.plot(cumulative_average[obs])
        obs_seen.append(obs)
plt.xscale('log')
#plt.show()
plt.savefig(bandit_path+'/log_rewards.eps', format='eps')
plt.legend(['state '+ str(i) for i in range(len(obs_seen))])
plt.savefig(bandit_path+'/log_rewards_legend.eps', format='eps')

# plot moving average ctr linear
for obs in cumulative_average.keys():
    if sum(cumulative_average[obs]) != 0:
        plt.plot(cumulative_average[obs])
plt.savefig(bandit_path+'/rewards.eps', format='eps')
plt.legend(['state '+ str(i) for i in range(len(obs_seen))])
#plt.show()
plt.savefig(bandit_path+'/rewards_legend.eps', format='eps')

errors = 0

for episode in range(test_episodes):
    # agent = B_lineAgent()
    if random.choice([0, 1]) == 0:
        current_env = BL_env
        adversary = B_lineAgent
    else:
        current_env = RM_env
        adversary = RedMeanderAgent
    bandit_obs = np.array([], dtype=int)
    blue_obs = current_env.reset()

    for step in range(N_STEPS):
        bandit_obs = np.append(bandit_obs, blue_obs)
        blue_obs, rew, done, info = current_env.step(0)

    blue_obs_hashable = ''.join(str(bit) for bit in bandit_obs)
    action = np.argmax(Q[blue_obs_hashable])
    #prediction = statistics.mode(predicitions)
    prediction = str(action_map[action].__name__)
    if action_map[action] != adversary:
        errors += 1

    #print('Adversary: '+ str(adversary.__name__) + ', Predicted: '+ prediction)
    #prediction = statistics.mode(alt_predicitions)

    #print('Adversary: '+ str(adversary.__name__) + ', ALT Predicted: '+ prediction)
print(Q)
print('Total errors: {:d}, over episodes: {:d}, success rate of {:.4f}%'.format(errors, test_episodes, (1-errors/test_episodes)*100))


bandit_save = '/'.join([bandit_path, "bandit_controller_{}.pkl".format(N_EPISODES)])
reward_save = '/'.join([bandit_path, "/bandit_controller_rewards_{}.pkl".format(N_EPISODES)])
with open(bandit_save, "wb") as bandit_save_file:  # Must open file in binary mode for pickle
    pkl.dump(Q, bandit_save_file)
with open(reward_save, "wb") as reward_save_file:  # Must open file in binary mode for pickle
    pkl.dump(rewards, reward_save_file)


