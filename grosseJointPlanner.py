import numpy as np
import itertools

"""
Creates a determinsitic transition table for a set of states and actions. If the action takes the agent off the board, the action should result in the next state being the same 
as the current state. If agents collide, one at random will move and the other will not.

Inputs:
    state set - list of states as tuple of tuples (single agent)
    action set - list of actions as tuple of tuples (single agent)
Output: nested dictionary {state:{action:nextState:probability}} where there is every state and action pair and only the next states that result in a non-zero probability
Once one agent is on the goal state, the transition moves to the terminal state no matter what actions are taken
Once the terminal state is reached, the next state will also always be the terminal state
"""

class SetupDeterministicTransitionByStateSet2Agent(object):
    def __init__(self, stateSet, actionSet, goalState):
        self.stateSet = stateSet
        # create a joint state set from a single agent state set, add terminal state to the set
        self.jointStateSet = [(s1, s2) for s1, s2 in itertools.product(stateSet, stateSet) if s1 != s2] + ['terminal'] 
        self.jointActionSet = list(itertools.product(actionSet, actionSet))
        self.goalState = goalState

    def __call__(self):
        transitionTable = {state: self.getStateTransition(state) for state in self.jointStateSet}
        return(transitionTable) 

    def getStateTransition(self, state):
        actionTransitionDistribution = {action: self.getStateActionTransition(state, action) for action in self.jointActionSet}
        return(actionTransitionDistribution)
    
    def getStateActionTransition(self, currentState, action):
        if currentState == 'terminal' or self.goalState in currentState:
            transitionDistribution = {'terminal': 1.0}
        else:
            transitionDistribution = self.getTransitionDistribution(currentState, action)
        return(transitionDistribution)

    def getTransitionDistribution(self, state, action):
        # if you directly apply the action to the current state, what the potential next state for each agent is
        potentialNextState = tuple([self.addTuples(agentS, agentA) for agentS, agentA in zip(state, action)])
        agent1NextState = potentialNextState[0]
        agent2NextState = potentialNextState[1]

        agent1Fixed = False
        agent2Fixed = False
        #if a move takes you off the board, you cannot take it and instead that agent remains stationary, if fixed = true, that agent must remain stationary
        if agent1NextState not in self.stateSet:
            agent1NextState = state[0]
            agent1Fixed = True
        if agent2NextState not in self.stateSet:
            agent2NextState = state[1]
            agent2Fixed = True

        # resulting joint state from taking into account moves off the board - is it a viable move
        onBoardPotentialNextState = (agent1NextState, agent2NextState)

        #if it is viable, agents will not collide and it should be in the joint state set
        if onBoardPotentialNextState in self.jointStateSet: 
            return({onBoardPotentialNextState:1.0})

        # if it is not in the joint state set, there is a collision
        if agent1NextState == agent2NextState:
            # collision 1: one agent runs into the stationary other
            if action[0] == (0,0) or action[1] == (0,0):
                return({state : 1.0})
            #collision 2: one agent tries to move off the board (and must stay stationary), the other collides into it there
            elif agent1Fixed or agent2Fixed:
                return({state:1.0})
            # collision 3: a collision on the board, probabilistically sample who moves and who stays
            else:
                agent1Moves = (agent1NextState, state[1])
                agent2Moves= (state[0], agent2NextState)
                return({agent1Moves: .5, agent2Moves: .5})
        
    
    def addTuples(self, tuple1, tuple2):
        lengthOfShorterTuple = min(len(tuple1), len(tuple2))
        summedTuple = tuple([tuple1[i] + tuple2[i] for i in range(lengthOfShorterTuple)])
        return(summedTuple)
        

"""
Reward table - 
Inputs:
    Constructor
    state set - list of states as tuple of tuples (single agent)
    action set - list of actions as tuple of tuples (single agent)

    Callable
    reward of goal state(s)
    cost of trap state(s)
    cost of taking action (0,0) - no movement 
Output: Nested dictionary of joint state, joint action, cost/reward
"""

class SetupRewardTable2AgentDistanceCost(object):
    def __init__(self, transitionTable, goalStates = [], trapStates = []):
        self.transitionTable = transitionTable
        self.goalStates = goalStates
        self.trapStates = trapStates
        
    def __call__(self, goalReward = 10, trapCost = -100, costOfNoMovement = .1):
        rewardTable = {state:{action: {nextState: self.applyRewardFunction(state, action, nextState, goalReward, trapCost, costOfNoMovement) \
                                        for nextState in nextStateDict.keys()} 
                                for action, nextStateDict in actionDict.items()} 
                        for state, actionDict in self.transitionTable.items()}
        return(rewardTable)

    def applyRewardFunction(self, state, action, nextState, goalReward, trapCost, costOfNoMovement):
        # terminal state has no reward or cost
        if state == 'terminal':
            return(0)
        #Unless already in the terminal state, incur the cost of action
        movementCosts = self.getCosts(state, action, costOfNoMovement)
        
        # if the intended next state is a special tile, 
        # the cost/reward of s, a, s' corresponds to the value of that tile
        specialTileCosts = self.getSpecialTileRewards(nextState, trapCost, goalReward)
        return(movementCosts+specialTileCosts)

    def getSpecialTileRewards(self, state, trapCost, goalReward):
        # if the next state is a special tile, the agent receives the rewards/costs of that location
        agent1State = state[0]
        agent2State = state[1]
        #get special rewards -- if either state is the goal state or the trap state
        reward = 0

        if (agent1State in self.goalStates) or (agent2State in self.goalStates):
            reward = reward + abs(goalReward)
        if agent1State in self.trapStates or agent2State in self.trapStates:
            reward = reward - abs(trapCost)
        return(reward)

    def getCosts(self, state, action, costOfNoMovement):
        # move costs - if in the goal state, no move cost 
        # because transition will move agent to terminal state no matter what
        agent1State = state[0]
        agent2State = state[1]
        if agent1State in self.goalStates or agent2State in self.goalStates:
            moveCost = 0
        else:
            moveCost = sum([self.getCostOfDistance(agentAction, costOfNoMovement) 
                for agentAction in action])
        return(moveCost)

    def getCostOfDistance(self, action, costOfNoMovement, nullAction = (0,0)):
        #Need to fix this for two agents
        if action == nullAction:
            return(-abs(costOfNoMovement))
        else:
            actionDistance = sum([abs(action[i]) for i in range(len(action))])
            return(-actionDistance)






"""
Reward table - 
Inputs:
    Constructor
    transition table
    goal state(s) - list of tuples
    trap state(s) - list of tuples
    goalReward - reward of goal state(s)
    trapCost - cost of trap state(s)
    costOfNoMovement - cost of taking action (0,0) - no movement 


    Callable
        agent abilities - tuple (agent1's ability, agetn 2's ability)
            with each element as numberic (0, infinity) values. This factor is multiplicative on the agent cost
            An ability less than 1 indicate stronger than baseline agent (costs become smaller),
            1 is costs = distances and larger than 1 are weaker agents (costs become larger)

Output: Dictionary of joint state, joint action, cost/reward
"""

class SetupRewardTable2AgentWeakStrong(object):
    def __init__(self, transitionTable, goalStates = [], trapStates = [], goalReward = 10, trapCost = -100, costOfNoMovement = .1):
        self.transitionTable = transitionTable
        self.goalStates = goalStates
        self.trapStates = trapStates

        self.goalReward = goalReward
        self.trapCost = trapCost
        self.costOfNoMovement = costOfNoMovement

        
    def __call__(self, agentAbilities):
        rewardTable = {state:{action: {nextState: self.applyRewardFunction(state, action, nextState, agentAbilities) \
                                        for nextState in nextStateDict.keys()} 
                                for action, nextStateDict in actionDict.items()} 
                        for state, actionDict in self.transitionTable.items()}
        return(rewardTable)

    def applyRewardFunction(self, state, action, nextState, agentAbilities):
        # terminal state has no reward or cost
        if state == 'terminal':
            return(0)
        #Unless already in the terminal state, incur the cost of action
        movementCosts = self.getCosts(state, action, agentAbilities)
        
        # if the intended next state is a special tile, 
        # the cost/reward of s, a, s' corresponds to the value of that tile
        specialTileCosts = self.getSpecialTileRewards(nextState)
        return(movementCosts+specialTileCosts)

    def getSpecialTileRewards(self, state):
        # if the next state is a special tile, the agent receives the rewards/costs of that location
        agent1State = state[0]
        agent2State = state[1]
        #get special rewards -- if either state is the goal state or the trap state
        reward = 0

        if (agent1State in self.goalStates) or (agent2State in self.goalStates):
            reward = reward + abs(self.goalReward)
        if agent1State in self.trapStates or agent2State in self.trapStates:
            reward = reward - abs(self.trapCost)
        return(reward)

    def getCosts(self, state, action, agentAbilities):
        # move costs - if in the goal state, no move cost 
        # because transition will move agent to terminal state no matter what
        agent1State = state[0]
        agent2State = state[1]
        if agent1State in self.goalStates or agent2State in self.goalStates:
            moveCost = 0
        else:
            moveCost = sum([self.getCostOfDistance(agentAction)*agentAbility for agentAction, agentAbility in zip(action, agentAbilities)])
        return(moveCost)

    def getCostOfDistance(self, action, nullAction = (0,0)):
        #Need to fix this for two agents
        if action == nullAction:
            return(-abs(self.costOfNoMovement))
        else:
            actionDistance = sum([abs(action[i]) for i in range(len(action))])
            return(-actionDistance)
