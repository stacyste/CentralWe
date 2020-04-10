import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from visualize.drawOnline import TransformCoord, DrawGrid, DrawRewards, DrawCircles, DrawShade, DrawTrajectoryColor, DrawPolicyArrows, Display
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
    gridNumberX = 5
    gridNumberY = 5
    states = list(itertools.product(range(gridNumberX), range(gridNumberY)))
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

    goals = [(0, 2), (4,0)]
    gettransition = SetupDeterministicTransitionByStateSet2Agent(states, actions, goals)
    transitionTable = gettransition()

    getReward = SetupRewardTable2AgentDistanceCost(transitionTable, goals)
    rewardTable = getReward()

    convergence = .000001
    gamma = .95
    valueTable = {state: 0 for state in transitionTable.keys()}
    beta = 3

    performValueIteration = BoltzmannValueIteration(transitionTable, rewardTable, valueTable, convergence, gamma, beta)
    optimalValues, policy = performValueIteration()

    initialState = ((0, 0), (4, 4))
    # print(policy[initialState])

    trajectory = samplePathToGoal(initialState, policy, transitionTable, goals)

    trajectoryToDraw = [[np.array(location) + (1,1) for location in timeStepTraj] for timeStepTraj in trajectory]
    # print(trajectoryToDraw)

    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    agentsColor = [GREEN, RED]

    fullScreen = False
    screenWidth = 800
    screenHeight = 800
    screen = initializeScreen(fullScreen, screenWidth, screenHeight)

    gridPixelSize = min(screenHeight // gridNumberX, screenWidth // gridNumberY)
    drawGrid = DrawGrid(screen, (gridNumberX, gridNumberY), gridPixelSize)

    currentDir = os.getcwd()
    rewardIconPath = os.path.join(currentDir, 'visualize', 'star.png')
    drawRewards = DrawRewards(screen, gridPixelSize, rewardIconPath)

    centerTransformCoord = TransformCoord(gridPixelSize, location='center')
    pointsWidth = (10, 20)
    drawCircles = DrawCircles(screen, centerTransformCoord, agentsColor, pointsWidth)

    leftTransformCoord = TransformCoord(gridPixelSize, location='left')
    drawShade = DrawShade(screen, leftTransformCoord, gridPixelSize)
    drawTrajectoryColor = DrawTrajectoryColor(drawShade, agentsColor)

    arrowUnitSize = gridPixelSize // 2
    drawPolicyArrows = DrawPolicyArrows(screen, centerTransformCoord, arrowUnitSize, actions)

    display = Display(screen, drawGrid, drawRewards, drawCircles, drawTrajectoryColor, drawPolicyArrows)

    dataIndex = 4
    imageFolderName = 'IWDetTransition'+ str(len(goals)) + 'GoalNoBarrier' + str(dataIndex) + 'beta' + str(beta)

    saveImageDir = os.path.join(os.path.join(currentDir, 'demo'), imageFolderName)
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)

    display(goals, trajectory, policy, saveImageDir, saveImage = True)


if __name__ == '__main__':
    main()


# new demos with new betas
# agent cost different

# movement cost + pushing cost -> reward table
# state = A1*A2*Object = (grid*grid)^3
# modify transition table