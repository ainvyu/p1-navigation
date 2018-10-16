# Banana Navigation

![banana](images/play.gif)

## Project details

This project contains a solution to the first project of Udacity Deep Reinforcement Learning.

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

It is helpful to check the repository below for details.
* https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation

## Getting started
### Prerequired
* Python 3.6
* Unity

And then to install python dependencies. 

    pip install -r requirements.txt

Then you should be able to run `jupyter notebook` and view `Navigation.ipynb`. 

The code for the Model and Agent are in `model.py` and `agent.py`, respectively.

## Instructions

Run each cell of `Navagation.ipynb`.

You can also run `run.py` to pop up the Unity Agent directly and check the behavior with the already trained weight.

    python run.py