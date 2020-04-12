import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from visualize.drawOnline import TransformCoord, DrawGrid, DrawObject, DrawCircles, DrawShade, DrawTrajectoryColor, DrawPolicyArrows, Display
from visualize.initialization import initializeScreen

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


from src.grosseJointPlanner import *
from src.ValueIteration import BoltzmannValueIteration


def randomSample(dist):
    targets = list(dist.keys())
    probOfTargets = [dist[target] for target in targets]
    sampleIndex = np.random.choice(len(targets), 1, p= probOfTargets)
    sampledTarget = targets[int(sampleIndex)]
    return sampledTarget


def samplePathToGoal(position, policy, transition, goals):
    trajectory = [position]

    while position[0] not in goals and position[1] not in goals:
        # take action probabilisitically
        sampledAction = randomSample(policy[position])

        # get new position -- Lucy Changed: able to handle probabilistic transition
        sampledNextState = randomSample(transition[position][sampledAction])

        # update to new belief/position and add to trajectory
        position = sampledNextState
        trajectory.append(position)

    return trajectory


def main():
    for i in range (10):

        # conditions:

        # 1-1:
        # goals = [(0, 2), (1, 3)]
        # agentAbilities = [1, 1] #  = agentActionCosts for each step
        # blocks = []
        # initialState = ((0, 0), (4, 4))

        # 1-2:
        # goals = [(2, 1), (4, 0)]
        # agentAbilities = [1, 1] #  = agentActionCosts for each step
        # blocks = []
        # initialState = ((0, 0), (4, 4))

        # 2-1:
        # goals = [(0, 2), (1, 3)]
        # agentAbilities = [1, 1] #  = agentActionCosts for each step
        # blocks = []
        # initialState = ((0, 0), (2, 4))

        # 2-2:
        # goals = [(2, 1), (4, 0)]
        # agentAbilities = [1, 1] #  = agentActionCosts for each step
        # blocks = []
        # initialState = ((0, 0), (3, 3))

        # 3-1:
        # goals = [(0, 2), (1, 3)]
        # agentAbilities = [3, 1] #  = agentActionCosts for each step
        # blocks = []
        # initialState = ((0, 0), (4, 4))

        # 3-2:
        # goals = [(2, 1), (4, 0)]
        # agentAbilities = [2, 1] #  = agentActionCosts for each step
        # blocks = []
        # initialState = ((0, 0), (4, 4))

        # 4-1:
        # goals = [(0, 2), (1, 3)]
        # agentAbilities = [1, 1] #  = agentActionCosts for each step
        # blocks = [(0, 1), (1, 1)]
        # initialState = ((0, 0), (4, 4))

        # 4-2:
        # goals = [(2, 1), (4, 0)]
        # agentAbilities = [2, 1] #  = agentActionCosts for each step
        # blocks = [(2, 0), (1, 1)]
        # initialState = ((0, 0), (4, 4))

        # 5-1-1:
        # goals = [(0, 2), (1, 3)]
        # agentAbilities = [1, 1] #  = agentActionCosts for each step
        # blocks = []
        # initialState = ((0, 0), (4, 4))
        # puddles = [(0, 1)]

        # 5-1-2:
        # goals = [(0, 2), (1, 3)]
        # agentAbilities = [1, 1] #  = agentActionCosts for each step
        # blocks = []
        # initialState = ((0, 0), (4, 4))
        # puddles = [(0, 1), (1, 0)]

        # 5-2:
        goals = [(2, 1), (4, 0)]
        agentAbilities = [1, 1] #  = agentActionCosts for each step
        blocks = []
        initialState = ((0, 0), (4, 4))
        puddles = [(1, 1), (1, 0)]

    ##
        gridNumberX = 5
        gridNumberY = 5
        states = list(itertools.product(range(gridNumberX), range(gridNumberY)))
        actions = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

        gettransition = SetupProbTransitionWithPuddle2Agent(states, actions, goals, puddles)
        transitionTable = gettransition()

        getReward = SetupRewardTable2AgentWeakStrong(transitionTable, goals, blocks)
        rewardTable = getReward(agentAbilities)

        convergence = .000001
        gamma = .95
        valueTable = {state: 0 for state in transitionTable.keys()}
        beta = 3

        performValueIteration = BoltzmannValueIteration(transitionTable, rewardTable, valueTable, convergence, gamma, beta)
        optimalValues, policy = performValueIteration()

        trajectory = samplePathToGoal(initialState, policy, transitionTable, goals)

        GREEN = (62, 96, 27)
        BLUE = (120, 205, 225)
        agentsColor = [GREEN, BLUE]

        fullScreen = False
        screenWidth = 800
        screenHeight = 800
        screen = initializeScreen(fullScreen, screenWidth, screenHeight)

        gridPixelSize = min(screenHeight // gridNumberX, screenWidth // gridNumberY)
        drawGrid = DrawGrid(screen, (gridNumberX, gridNumberY), gridPixelSize)

        currentDir = os.getcwd()
        rewardIconPath = os.path.join(currentDir, 'visualize', 'star.png')
        drawRewards = DrawObject(screen, gridPixelSize, rewardIconPath)

        blockIconPath = os.path.join(currentDir, 'visualize', 'block.png')
        drawBlock = DrawObject(screen, gridPixelSize, blockIconPath)

        puddleIconPath = os.path.join(currentDir, 'visualize', 'water.png')
        drawPuddles = DrawObject(screen, gridPixelSize, puddleIconPath)

        centerTransformCoord = TransformCoord(gridPixelSize, location='center')
        pointsWidth = (10, 10)
        drawCircles = DrawCircles(screen, centerTransformCoord, agentsColor, pointsWidth)

        leftTransformCoord = TransformCoord(gridPixelSize, location='left')
        drawShade = DrawShade(screen, leftTransformCoord, gridPixelSize)
        drawTrajectoryColor = DrawTrajectoryColor(drawShade, agentsColor)

        arrowUnitSize = gridPixelSize // 2
        drawPolicyArrows = DrawPolicyArrows(screen, centerTransformCoord, arrowUnitSize, actions)

        display = Display(screen, drawGrid, drawRewards, drawBlock, drawPuddles, drawCircles, drawTrajectoryColor, drawPolicyArrows)


        dataIndex = i
        imageFolderName = 'IWDemo' + str(dataIndex) + 'beta' + str(beta)

        saveImageDir = os.path.join(os.path.join(currentDir, 'demo'), imageFolderName)
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)

        display(goals, blocks, puddles, trajectory, policy, saveImageDir, saveImage = True)


if __name__ == '__main__':
    main()


# new demos with new betas
# agent cost different

# movement cost + pushing cost -> reward table
# state = A1*A2*Object = (grid*grid)^3
# modify transition table