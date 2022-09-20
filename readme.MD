![mindrake](https://user-images.githubusercontent.com/10000317/150498045-b712992c-b569-4654-a35e-65660df3f795.png)

# Team Mindrake
*"For death or glory"*

This repository contains the model that won the 3rd place in [CAGE-Challenge-2](https://github.com/cage-challenge/cage-challenge-2). This model has shown clear performance improvement under CAGE-Challenge-2 CybORG environment, compared with our original [winning model](https://github.com/alan-turing-institute/cage-challenge-1/blob/submission_final) in [CAGE-Challenge-1](https://github.com/cage-challenge/cage-challenge-1) under the same environment. 

<br><br/>

# Model Architecture
Our blue agent here keeps the hierarchical structure, meaning it has a controller sitting on top of two subagents, and each subagent is specialilsed at defending against one type of attacker. The controller receives the obervations of the network at beginning of each episode and pick a specialised subagent to defend the network. Subagents are pretrained Proximal Policy Optimisation (PPO) reinforcement learning agents, which have formed converged policies to defend against their corresponding attackers. 

<!-- There are two types of controller developed, *Bandit* and *Heuristic*, both can achive 100% accuracy when choosing the subagent. Heuristic controller makes use of the difference in the behaviour patterns of the first 4 steps to dinstiguish attackers, hence it doesn't require training. Whereas Bandit controller uses simple bandit learning algorithm which has been pretrained for 15000 steps.  -->

The controller can achive 100% accuracy when choosing the subagent. It uses simple bandit learning algorithm which has been pretrained for 15000 steps.

The attackers are **MeanderAgent** (has no information about the network so it attacks the hosts at random) and **BLineAgent** (has information about the network so it has clear strategy to exploit operational server). Subagent for **MeaderAgent** uses PPO algorithms and 52-bit observation space, while subagent for **BLineAgent** uses PPO with curiosity and 27-float observation space. 

<br><br/>

# What is in the repo
### There are two folders in the main directory: <br>

`agents/baseline_sub_agents/` -- contains the scripts to load both types of controllers and subagents; <br>

* `evaluation.py` can evaluate the hierarchical model 
* `loadBanditController.py` can retrieve the pretrained controller and subagents, which is used by `evaluation.py`
* BlineAgent defender uses  `bline_CybORGAgent.py` to setup the environment; `StateRepWrapper.py` and `newBlueTableWrapper.py` are used to create the 27-float observation space. `curiosity.py` is used to add curiosity in the RL algorithm
* MeanderAgent defender uses `CybORGAgent.py` as the environment, where `ChallengeWrapper` creates 52-bit observation space. 
* `configs.py` contrains RL configurations when training both subagents
* `neural_nets.py` includes the customised neural network used in subagents
* `train_simple_bandit.py` is used to train the bandit controller
* `train_subagent.py` is used to train the subagents


`logs/` -- contains the pretrained controller and subagent models.

* `bandits/` contains pretrained bandit controller (i.e. `bandit_controller_15000.pkl`)
* `various/` contains pretrained **MeanderAgent** defender (`PPO_RedMeanderAgent_2022-07-06_16-32-36`) and **BLineAgent** defender (`SR_B_lineAgent_new52obs-27floats_2022-07-16_16-40-09`)

<br>

### CAGE Challenge 2 submission from Team Mindrake.

Evaluation output file:
 - `20220719_103759_LoadBanditBlueAgent.txt`

Terminal Ouptut file:
 - `terminal_output.txt`

Evaluation Script:
 - `/agents/baseline_sub_agents/evaluation.py`

<br><br/>

# How to run the code
## Setup and installation
### Install [CAGE Challenge](https://github.com/cage-challenge/cage-challenge-2)

```
# Grab the repo
git clone https://github.com/cage-challenge/cage-challenge-2.git

# from the cage-challenge-2/CybORG directory
pip install -e .
```

### Install our requirements

```
pip install -r requirements.txt
```
<br>

## Model training
To train subagnts
```
# assume you are in the main directory
cd agents/baseline_sub_agents/

# to train BLineAgent defender
python train_subagent.py bline

# to train MeanderAgent defender
python train_subagent.py meander

# to train bandit controller
python train_simple_bandit.py
```

## Model evaluation
```
# assume you are in the main directory
cd agents/baseline_sub_agents/
```


### if you are using the pretrained models we provided:
```
python evaluation.py
```

### if you want to use the models you trained yourself:
- Change the model directory in `sub_agents.py`
- `python evaluation.py`

<br>



