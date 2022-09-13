![mindrake](https://user-images.githubusercontent.com/10000317/150498045-b712992c-b569-4654-a35e-65660df3f795.png)

# Team Mindrake
*"For death or glory"*

This repository contains the model that won the 3rd place in [CAGE-Challenge-2](https://github.com/cage-challenge/cage-challenge-2). This model has shown clear imporvement in the performance under CAGE-Challenge-2 CybORG environment, compared with our [winning model](https://github.com/alan-turing-institute/cage-challenge-1/blob/submission_final) in [CAGE-Challenge-1](https://github.com/cage-challenge/cage-challenge-1) under the same environment. 


<br><br/>

# Model Architecture
Our blue agent here keeps the hierarchical structure, meaning it has a controller sitting on top of two subagents, and each subagent is specialilsed at defending against one type of attacker. The controller received the obervations of the network at beginning of each episode and pick a specialised subagent to defend the network for each episode. Subagents are pretrained Proximal Policy Optimisation (PPO) reinforcement learning agents, which have formed converged policies to defend against their corresponding attackers. 

There are two types of controller developed, *Bandit* and *Heuristic*, both can achive 100% accuracy when choosing the subagent. Heuristic controller makes use of the difference in the behaviour patterns of the first 4 steps to dinstiguish attackers, hence it doesn't require training. Whereas Bandit controller uses simple bandit learning algorithm which has been pretrained for 15000 steps. 

The attackers are **MeanderAgent** (has no information about the network so it attacks the hosts at random) and **BLineAgent** (has information about the network so it has clear strategy to exploit operational server). Subagent for **MeaderAgent** uses PPO algorithms and 52bits observation space. Subagent for **BLineAgent** uses PPO with curiosity and 27floats observation space. 

<br><br/>

# What is in the repo
There are two folder in the main directory: <br>

`agents/` -- contains the scripts to load both types of controllers and subagents; <br>
<ul>
    <li> 
    <li>
</ul>


`logs/` -- contains the pretrained controller and subagent models.



<br><br/>

# How to run the code
## Setup and installation
## Model evaluation



