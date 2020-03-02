#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import sys
import pandas as pd


# In[2]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[3]:

from grosseJointPlanner import *
from ValueIteration import BoltzmannValueIteration


# In[4]:


gridWidth = 5
gridHeight = 5
states = list(itertools.product(range(gridWidth), range(gridHeight)))
actions = [(-1,0), (0,1), (1,0), (0,-1), (0,0)]


# In[5]:


gettransition = SetupDeterministicTransitionByStateSet2Agent(states, actions, (1,1))
transitionTable = gettransition()
getReward = SetupRewardTable2AgentDistanceCost(transitionTable, [(4,2)])


# In[6]:


transitionTable = gettransition()
rewardTable = getReward()


# In[9]:


convergence = .000001
gamma = .95
valueTable = {state:0 for state in transitionTable.keys()}
beta = 2

performValueIteration = BoltzmannValueIteration(transitionTable, rewardTable, valueTable, convergence, gamma, beta)
optimalValues, policy = performValueIteration()


# In[27]:


#Trajectory sampling
def samplePathToGoal(position, policy, transition, goals):
    trajectory = [position]

    while position[0] not in goals and position[1] not in goals:
        
        #take action probabilisitically
        actions = list(policy[position].keys())
        probOfAction = [policy[position][action] for action in actions]
        actionIndex = np.random.choice(len(actions), 1, p = probOfAction)
        sampledAction = actions[int(actionIndex)]       
        
        #get new position
        newPosition = list(transition[position][sampledAction].keys())[0]

        #update to new belief/position and add to trajectory
        position = newPosition
        trajectory.append(position)
    return(trajectory)



trajectory = samplePathToGoal(((0, 0), (4, 4)), policy, transitionTable, [(4,2)])

