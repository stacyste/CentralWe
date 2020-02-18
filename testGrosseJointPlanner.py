import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import grosseJointPlanner as targetCode
import pandas as pd
import itertools

@ddt
class TestTransitionByStateSet2Agent(unittest.TestCase):
	def setUp(self): 
		self.cardinalActionSet = [(-1,0), (0,1), (1,0), (0,-1), (0,0)]
		gridWidth = 4
		gridHeight = 4
		self.stateSet4x4 = list(itertools.product(range(gridWidth), range(gridHeight)))
		self.goalState = (3,3)
		self.trapState = (0,0)

	#create a deterministic transition table for the joint next state of two agents' actions in current state
	#resolves collisions by unif sampling who moves
	# examples 1, 2, 3, 4 treat two agent valid moves on board
	# examples 5, 6 treat agent 1 valid move and agent 2 move off board
	# examples 7, 8 treat agent 2 valid move and agent 1 move off board
	# example 9 treat both move off board
	# examples 10,11 treat collision onto a valid tile
	# examples 12, 13 are collisions where one agent runs into the stationary other
	# examples 14, 15 reach terminal as the next state
	#examples 16, 17, 18 are from terminal
	@data((((0,0), (1,0)), ((0,0), (0,0)), {((0,0), (1,0)):1.0}), 
		(((0,0), (1,0)), ((0,1), (0,1)), {((0,1), (1,1)):1.0}), 
		(((2,2), (1,0)), ((0,-1), (0,1)), {((2,1), (1,1)):1.0}), 
		(((0,0), (1,0)), ((1,0), (-1,0)), {((1,0), (0,0)):1.0}),

		(((0,0), (1,0)), ((0,0), (0,-1)), {((0,0), (1,0)):1.0}), 
		(((0,0), (1,0)), ((0,1), (0,-1)), {((0,1), (1,0)):1.0}), 

		(((0,0), (1,0)), ((-1,0), (0,1)), {((0,0), (1,1)):1.0}), 
		(((0,1), (1,1)), ((-1,0), (0,-1)), {((0,1), (1,0)):1.0}),

		(((0,0), (1,0)), ((-1,0), (0,-1)), {((0,0), (1,0)):1.0}),

		(((1,0), (0,1)), ((0,1), (1,0)), {((1,1), (0,1)):.5, ((1,0), (1,1)):.5}),
		(((0,0), (1,1)), ((0,1), (-1,0)), {((0,0), (0,1)): .5, ((0,1), (1,1)):.5}),

		(((1,0), (1,1)), ((0,1), (0,0)), {((1,0), (1,1)):1.0}),
		(((1,0), (1,1)), ((0,0), (0,-1)), {((1,0), (1,1)):1.0}),

		(((3,3), (1,0)), ((-1,0), (-1,0)), {'terminal':1.0}),
		(((0,0), (3,3)), ((0,0), (0,0)), {'terminal':1.0}),

		('terminal', ((0,0), (1,0)), {'terminal':1.0}),
		('terminal', ((1,0), (0,1)), {'terminal':1.0}),
		('terminal', ((0,0), (0,0)), {'terminal':1.0}) )
	@unpack
	def test_SetupDeterministicTransitionByStateSet2Agent(self, jointState, jointAction, expectedResult):
		getTransitionTable = targetCode.SetupDeterministicTransitionByStateSet2Agent(self.stateSet4x4,self.cardinalActionSet, self.goalState)
		transitionTable = getTransitionTable()
		nextStateDistribution = transitionTable[jointState][jointAction]

		self.assertEqual(nextStateDistribution, expectedResult)


	def tearDown(self):
		pass

@ddt
class TestRewardSetup(unittest.TestCase):
	def setUp(self): 
		cardinalActionSet = [(-1,0), (0,1), (1,0), (0,-1), (0,0)]
		gridWidth = 4
		gridHeight = 4
		self.goalState = (3,3)
		self.trapState = (0,0)
		stateSet4x4 = list(itertools.product(range(gridWidth), range(gridHeight)))
		getTransitionTable = targetCode.SetupDeterministicTransitionByStateSet2Agent(stateSet4x4,cardinalActionSet, self.goalState)
		self.transitionTable = getTransitionTable()

	# examples 1, 2, 3 are standard both agent movements
	# examples 4, 5, 6 are each agent and both agents staying still
	# examples 7, 8, 9, 10 are at the goal state -> terminal
	# examples 11, 12, 13 are at the terminal
	# examples 14, 15, 16 are collisions or moves off the board 
	# examples 17,18, 19 are at trap states - 
	# 1 on trap staying, 2 moving; 1 moving, 2 on trap moving; 1 staying, 2 on trap moving; 
	# examples 20, 21are moving on trap and reward
	# examples 22, 23 are moving onto the reward
	@data((((0,1), (1,0)), ((0, 1), (1,0)), ((0, 2), (2,0)), -2), 
		(((2,2), (3,2)), ((1, 0), (0,-1)), ((3, 2), (3,1)), -2), 
		(((3,2), (1,2)), ((-1, 0), (-1,0)), ((2, 2), (0,2)), -2),

		(((0,1), (1,0)), ((0, 0), (1,0)), ((0, 1), (2,0)), -1.1), 
		(((2,2), (1,0)), ((0, -1), (0,0)), ((2, 1), (1,0)), -1.1), 
		(((2,1), (1,0)), ((0, 0), (0,0)), ((2, 1), (1,0)), -.2),

		(((3,3), (1,0)), ((0, 1), (1,0)), 'terminal', 0), 
		(((3,3), (1,0)), ((0, 1), (1,0)), 'terminal', 0), 
		(((0,0), (3,3)), ((0, 1), (1,0)), 'terminal', 0), 
		(((0,1), (3,3)), ((0, -1), (1,0)), 'terminal', 0), 

		('terminal', ((0,1), (-1, 0)),'terminal', 0),
		('terminal', ((1,0), (0, 0)),'terminal', 0),
		('terminal', ((0,0), (0, 1)),'terminal', 0),
		### check this group
		(((0,1), (1,0)), ((1, 0), (0,1)), ((1,1), (1,0)), -2), 
		(((0,1), (1,0)), ((-1, 0), (1,0)), ((0,1), (2,0)), -2), 
		(((0,1), (1,0)), ((0, 0), (0,-1)), ((0,1), (1,0)), -1.1),

		(((0,0), (0,1)), ((0, 0), (0,1)), ((0,0), (0,2)), -101.1),
		(((0,1), (1,0)), ((1, 0), (-1,0)), ((1,1), (0,0)), -102),
		(((2,1), (0,0)), ((0, 0), (0,0)), ((2,1), (0,0)), -100.2),

		(((3,2), (0,1)), ((0, 1), (0,-1)), ((3,3), (0,0)), -92), 
		(((3,2), (0,0)), ((0, 1), (0,0)), ((3,3), (0,0)), -91.1), 

		(((2,1), (3,2)), ((0, 0), (0,1)), ((2,1), (3,3)), 8.9),
		(((2,0), (3,2)), ((0, 1), (0,1)), ((2,1), (3,3)), 8))
	@unpack
	def test_SetupRewardTable2AgentDistanceCost(self, jointState, jointAction, nextState, expectedResult):
		getRewardTable = targetCode.SetupRewardTable2AgentDistanceCost(self.transitionTable, [self.goalState], [self.trapState])
		rewardTable = getRewardTable()
		stateActionReward = rewardTable[jointState][jointAction][nextState]

		self.assertEqual(stateActionReward, expectedResult)
	
	def tearDown(self):
		pass

@ddt
class TestRewardSetup_WeakStrongAgents(unittest.TestCase):
	def setUp(self): 
		cardinalActionSet = [(-1,0), (0,1), (1,0), (0,-1), (0,0)]
		gridWidth = 4
		gridHeight = 4
		self.goalState = (3,3)
		self.trapState = (0,0)
		stateSet4x4 = list(itertools.product(range(gridWidth), range(gridHeight)))
		getTransitionTable = targetCode.SetupDeterministicTransitionByStateSet2Agent(stateSet4x4,cardinalActionSet, self.goalState)
		self.transitionTable = getTransitionTable()

	#examples 1-8 for each category of previous example (1,1) ability
	# examples 9-16 for agent 1 as weaker than 1(>1), agent 2=1
	# examples 17-24 for agent 2 as stronger (<1), agent 1=1
	# examples 25-32 for agent 1 stronger and agent 2 weaker
	@data(((1,1),((0,1), (1,0)), ((0, 1), (1,0)), ((0, 2), (2,0)), -2),
		   ((1,1), ((0,1), (1,0)), ((0, 0), (1,0)), ((0, 1), (2,0)), -1.1),
		   ((1,1), ((3,3), (1,0)), ((0, 1), (1,0)), 'terminal', 0),
		   ((1,1), 'terminal', ((0,1), (-1, 0)),'terminal', 0),
		   ((1,1), ((0,1), (1,0)), ((1, 0), (0,1)), ((1,1), (1,0)), -2),
		   ((1,1), ((0,0), (0,1)), ((0, 0), (0,1)), ((0,0), (0,2)), -101.1),
		   ((1,1), ((3,2), (0,1)), ((0, 1), (0,-1)), ((3,3), (0,0)), -92),
		   ((1,1), ((2,1), (3,2)), ((0, 0), (0,1)), ((2,1), (3,3)), 8.9),

		   ((2,1),((0,1), (1,0)), ((0, 1), (1,0)), ((0, 2), (2,0)), -3), #typical no collision move
		   ((1.5,1), ((0,1), (1,0)), ((0, 0), (1,0)), ((0, 1), (2,0)), -1.1), #agent 1 doesn't move
		   ((12,1), ((3,3), (1,0)), ((0, 1), (1,0)), 'terminal', 0), #once on goal, dont incur costs
		   ((7,1), 'terminal', ((0,1), (-1, 0)),'terminal', 0), # terminal does not incur costs
		   ((5.5,1), ((0,1), (1,0)), ((1, 0), (0,1)), ((1,1), (1,0)), -6.5), #movements off board incur costs 
		   ((12,1), ((0,0), (0,1)), ((0, 0), (0,1)), ((0,0), (0,2)), -101.1), #trap state, but also stationary agent 1
		   ((12,1), ((3,2), (0,1)), ((0, 1), (0,-1)), ((3,3), (0,0)), -103), # trap cost, reward bonus + move cost
		   ((2,1), ((2,1), (3,2)), ((0, 0), (0,1)), ((2,1), (3,3)), 8.9) , # reward bonus, agent 1 no move cost

		   ((1,0.5),((0,1), (1,0)), ((0, 1), (1,0)), ((0, 2), (2,0)), -1.5), #typical movement
		   ((1,0.1), ((0,1), (1,0)), ((0, 0), (1,0)), ((0, 1), (2,0)), -.2), #typical movement
		   ((1,0.25), ((3,3), (1,0)), ((0, 1), (1,0)), 'terminal', 0), #on goal
		   ((1,0.4), 'terminal', ((0,1), (-1, 0)),'terminal', 0), #at terminal
		   ((1,0.5), ((0,1), (1,0)), ((1, 0), (0,1)), ((1,1), (1,0)), -1.5), #movements off board 
		   ((1,0.25), ((0,0), (0,1)), ((0, 0), (0,1)), ((0,0), (0,2)), -100.35), #agent 1 sits on trap
		   ((1,0.9), ((3,2), (0,1)), ((0, 1), (0,-1)), ((3,3), (0,0)), -91.9), #agent 2 moves onto trap while agent 1 to goal
		   ((1,0.1), ((2,1), (3,2)), ((0, 0), (0,1)), ((2,1), (3,3)), 9.8) , #agent 1 still,agent 2 onto goal

		   ((.5,1.1),((0,1), (1,0)), ((0, 1), (1,0)), ((0, 2), (2,0)), -1.6),
		   ((.4,1.5), ((0,1), (1,0)), ((0, 0), (1,0)), ((0, 1), (2,0)), -1.6),
		   ((.3,2), ((3,3), (1,0)), ((0, 1), (1,0)), 'terminal', 0),
		   ((.2,3), 'terminal', ((0,1), (-1, 0)),'terminal', 0),
		   ((.1,5), ((0,1), (1,0)), ((1, 0), (0,1)), ((1,1), (1,0)), -5.1),
		   ((.6,10), ((0,0), (0,1)), ((0, 0), (0,1)), ((0,0), (0,2)), -110.1),
		   ((.7,7), ((3,2), (0,1)), ((0, 1), (0,-1)), ((3,3), (0,0)), -97.7),
		   ((.8,8), ((2,1), (3,2)), ((0, 0), (0,1)), ((2,1), (3,3)), 1.9)  )
	@unpack
	def test_SetupRewardTableforWeakAndStrongAgents(self, agentAbilities, jointState, jointAction, nextState, expectedResult):
		getRewardTable = targetCode.SetupRewardTable2AgentWeakStrong(self.transitionTable, [self.goalState], [self.trapState])
		rewardTable = getRewardTable(agentAbilities)
		stateActionReward = rewardTable[jointState][jointAction][nextState]

		self.assertAlmostEqual(stateActionReward, expectedResult)
	
	def tearDown(self):
		pass


 
if __name__ == '__main__':
	unittest.main(verbosity=2)