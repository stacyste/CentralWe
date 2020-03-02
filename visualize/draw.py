import numpy as np
import sys
import pygame

class DrawCircles:
    def __init__(self, colorList, pointsWidth, pointExtendTime = 100, FPS = 60):
        self.FPS = FPS
        self.pointExtendFrame = int(pointExtendTime * self.FPS / 1000)
        self.pointsWidth = pointsWidth
        self.colorList = colorList
        self.drawPoint = lambda game, color, point, pointWidth: pygame.draw.circle(game, color, point, pointWidth)

    def __call__(self, screen, pointsLocation):
        for frameNumber in range(self.pointExtendFrame):
            for agentIndex in range(len(self.colorList)):
                self.drawPoint(screen, self.colorList[agentIndex], pointsLocation[agentIndex], self.pointsWidth[agentIndex])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()
        return screen




class DrawGrid:
    def __init__(self, game, gridSize, gridPixelSize, backgroundColor = (255, 255, 255), gridColor = (0,0,0), gridLineWidth = 3):
        self.game = game
        self.gridNumberX, self.gridNumberY = gridSize
        self.gridPixelSize = gridPixelSize
        self.backgroundColor = backgroundColor
        self.gridColor = gridColor
        self.gridLineWidth = gridLineWidth

    def __call__(self):
        upperBoundX = self.gridPixelSize * self.gridNumberX
        upperBoundY = self.gridPixelSize * self.gridNumberY
        self.game.fill(self.backgroundColor)

        for gridIndexX in range(self.gridNumberX + 1):
            gridX = int(gridIndexX * self.gridPixelSize)
            pygame.draw.line(self.game, self.gridColor, (gridX, 0), (gridX, upperBoundY), self.gridLineWidth)

        for gridIndexY in range(self.gridNumberY + 1):
            gridY = int(gridIndexY * self.gridPixelSize)
            pygame.draw.line(self.game, self.gridColor, (0, gridY), (upperBoundX, gridY), self.gridLineWidth)

        return self.game



class DrawCirclesAndLines:
    def __init__(self, modifyOverlappingPoints, pointExtendTime = 100, FPS = 60, circleSize = 10, lineColor = (0,0,0)):
        self.FPS = FPS
        self.pointExtendFrame = int(pointExtendTime * self.FPS / 1000)
        self.circleSize = circleSize
        self.modifyOverlappingPoints = modifyOverlappingPoints
        self.lineColor = lineColor
        self.drawPoint = lambda game, color, point: pygame.draw.circle(game, color, point, self.circleSize)

    def __call__(self, screen, pointsLocation, colorList, lineWidthDict):
        for frameNumber in range(self.pointExtendFrame):
            pointsLocation = self.modifyOverlappingPoints(pointsLocation)

            for agentIndex in range(len(colorList)):
                self.drawPoint(screen, colorList[agentIndex], pointsLocation[agentIndex])

            for lineConnectedAgents in lineWidthDict.keys():
                pullingAgentIndex, pulledAgentIndex = lineConnectedAgents
                pullingAgentLoc = pointsLocation[pullingAgentIndex]
                pulledAgentLoc = pointsLocation[pulledAgentIndex]
                lineWidth = lineWidthDict[lineConnectedAgents]
                pygame.draw.line(screen, self.lineColor, pullingAgentLoc, pulledAgentLoc, lineWidth)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.flip()
        return screen


class DrawPointsAndSaveImage:
    def __init__(self, drawGrid, drawReward, drawCircles, drawTrajectoryColor, drawPolicyArrows, gridPixelSize, fps = 60):
        self.fps = fps
        self.drawGrid = drawGrid
        self.drawReward = drawReward
        self.drawCircles = drawCircles
        self.drawTrajectoryColor = drawTrajectoryColor
        self.drawPolicyArrows = drawPolicyArrows

        self.gridPixelSize = gridPixelSize

    def __call__(self, rewardCoords, trajectoryToDraw, trajectory, policy, saveImageDir = None, saveImage = False):
        fpsClock = pygame.time.Clock()
        for timeStep in range(len(trajectoryToDraw)):
            state = trajectoryToDraw[timeStep]
            fpsClock.tick(self.fps)
            game = self.drawGrid()
            game = self.drawReward(rewardCoords)
            self.drawTrajectoryColor(game, trajectoryToDraw)
            self.drawPolicyArrows(game, trajectoryToDraw, trajectory, policy)
            pointsLocation = [list(np.array(agentState) * self.gridPixelSize - self.gridPixelSize//2)
                                  for agentState in state]
            game = self.drawCircles(game, pointsLocation)
            if saveImage:
                pygame.image.save(game, saveImageDir + '/' + format(timeStep, '04') + ".png")

        return

class DrawRewards:
    def __init__(self, game, gridPixelSize, rewardIconPath):
        self.game = game
        self.gridPixelSize = gridPixelSize
        self.rewardIconPath = rewardIconPath

    def __call__(self, rewardCoords):
        rewardSymbol = pygame.image.load(self.rewardIconPath)
        rewardSymbol = pygame.transform.scale(rewardSymbol, (self.gridPixelSize//2, self.gridPixelSize//2))
        rewardsLocation = [np.array(coord) * self.gridPixelSize + self.gridPixelSize//4 for coord in rewardCoords]
        for rewardCoord in rewardsLocation:
            self.game.blit(rewardSymbol, rewardCoord)
        return self.game



class TransformCoord:
    def __init__(self, gridSize, location, edgeSize = 3):
        self.gridSize = gridSize
        self.edgeSize = edgeSize
        self.location = location

    def __call__(self, coords, center=False):
        if self.location == 'center':  # make the coords the center of the shape
            xLoc = int((coords[0] - 1) * self.gridSize + self.edgeSize + self.gridSize / 2)
            yLoc = int((coords[1] - 1) * self.gridSize + self.edgeSize + self.gridSize / 2)
        else:
            xLoc = int((coords[0] - 1) * self.gridSize + self.edgeSize)
            yLoc = int((coords[1] - 1) * self.gridSize + self.edgeSize)

        return (xLoc, yLoc)


class DrawTrajectoryColor:
    def __init__(self, leftTransformCoord, boxColors, gridSize):
        self.leftTransformCoord = leftTransformCoord
        self.agentsColors = boxColors
        self.gridSize = gridSize

    def __call__(self, game, trajectoryToDraw):
        getUniqueStates = lambda trajectory: list(set([tuple(state) for state in trajectory]))
        agentNum = len(self.agentsColors)
        agentsTraj = [getUniqueStates([state[agentID] for state in trajectoryToDraw]) for agentID in range(agentNum)]

        fillSquare = pygame.Surface((self.gridSize, self.gridSize), pygame.SRCALPHA)
        for agentID in range(agentNum):
            agentCol = self.agentsColors[agentID]
            pygame.draw.rect(fillSquare, agentCol, fillSquare.get_rect())
            traj = agentsTraj[agentID]
            for state in traj:
                coord = self.leftTransformCoord(state)
                game.blit(fillSquare, coord)

        return game


class DrawPolicyArrows:
    def __init__(self, centerTransformCoord, arrowUnitSize, actionSpace, lineWidth = 3, lineCol = (0,0,0)):
        self.centerTransformCoord = centerTransformCoord
        self.arrowUnitSize = arrowUnitSize
        self.actionSpace = actionSpace
        self.lineWidth = lineWidth
        self.lineCol = lineCol

    def __call__(self, game, trajectoryToDraw, trajectory, policy):
        timeSpan = len(trajectory)-1
        agentNumber = len(trajectory[0])
        for time in range(timeSpan):
            currentState = trajectory[time]
            oneStateDist = policy[currentState]
            for agentID in range(agentNumber):
                arrowCenter = self.centerTransformCoord(np.array(currentState[agentID]) + 1)

                # actionDist = {action: np.sum([oneStateDist[actionPair] for actionPair in oneStateDist.keys() if actionPair[0] == action])
                #               for action in self.actionSpace}
                for action in self.actionSpace:
                    prob = np.sum([oneStateDist[actionPair] for actionPair in oneStateDist.keys() if actionPair[agentID] == action])
                    # if agentID == 1:
                    #     print(time)
                    #     print(action)
                    #     print(prob)
                    arrowLength = prob * self.arrowUnitSize
                    if action !=(0,0):
                        direction = np.array(action)/np.linalg.norm(action, ord=2)
                        endPos = (arrowCenter + arrowLength* direction).astype(int)
                        pygame.draw.line(game, self.lineCol, arrowCenter, endPos, self.lineWidth)
                        pygame.draw.circle(game, self.lineCol, endPos, int(2 * self.lineWidth))
                    else:
                        pygame.draw.circle(game, self.lineCol, arrowCenter, int(2 * self.lineWidth))
        return game



















