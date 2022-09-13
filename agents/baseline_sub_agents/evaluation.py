import subprocess
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent

from CybORG.Agents.Wrappers import ChallengeWrapper

from loadBanditController import LoadBanditBlueAgent as LoadBlueAgent
from CybORGActionAgent import CybORGActionAgent
MAX_EPS = 100
agent_name = 'Blue'

def wrap(env):
    return ChallengeWrapper(env=env, agent_name='Blue')

def custom_wrap(config):
    return CybORGActionAgent(config)

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    commit_hash = get_git_revision_hash()
    # ask for a name
    name = 'Myles Foley' #input('Name: ')
    # ask for a team
    team = 'Mindrake' #input("Team: ")
    # ask for a name for the agent
    name_of_agent = 'Test evaluation'#input("Name of technique: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    agent = LoadBlueAgent()

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:
            r_step = {i:[] for i in range(1, num_steps + 1)}
            #print(r_step)
            #print(type(list(r_step)[0]))

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            #wrapped_cyborg = ChallengeWrapper(env=cyborg, agent_name='Blue') #wrap(cyborg)

            wrapped_cyborg = custom_wrap({'agent_name': 'Blue', 'env':cyborg, 'max_steps': 100, 'attacker': red_agent})

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    #action, agent_selected = agent.get_action(observation, action_space)
                    action, agent_to_select = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    # result = cyborg.step(agent_name, action)

                    # Print true table on each step
                    #true_state = cyborg.get_agent_state('True')
                    #true_table = true_obs_to_table(true_state,cyborg)
                    #print(true_table)
                    if agent_to_select == 0:
                        agent_to_select = 'Meander'
                    else:
                        agent_to_select = 'BLine'
                    r.append(rew)
                    r_step[j+1].append(rew)
                    # r.append(result.reward)
                    #agent_selected = 'BLine' if agent_selected == 0 else 'RedMeander'
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red')), str(agent_to_select)))
                agent.end_episode()    # Don't forget to dangermouse
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            r_step = {step: mean(r_step[step]) for step in range(1, num_steps+1)}

            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            #print(sum(r_step.values()))
            #include average rewards per step
            #print(f'Average reward per step for red agent {red_agent.__name__} and steps {num_steps} is: ' +str(r_step))
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
