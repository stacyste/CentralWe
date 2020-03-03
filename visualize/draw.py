import numpy as np
import sys
import pygame

class DrawCircles:
    def __init__(self, screen, colorList, pointsWidth, pointExtendTime = 100, FPS = 60):
        self.screen = screen
        self.FPS = FPS
        self.pointExtendFrame = int(pointExtendTime * self.FPS / 1000)
        self.pointsWidth = pointsWidth
        self.colorList = colorList
        self.drawPoint = lambda game, color, point, pointWidth: pygame.draw.circle(game, color, point, pointWidth)

    def __call__(self, pointsLocation):
        for frameNumber in range(self.pointExtendFrame):
            for agentIndex in range(len(self.colorList)):
                self.drawPoint(self.screen, self.colorList[agentIndex], pointsLocation[agentIndex], self.pointsWidth[agentIndex])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()


class DrawGrid:
    def __init__(self, screen, gridSize, gridPixelSize, backgroundColor = (255, 255, 255), gridColor = (0,0,0), gridLineWidth = 3):
        self.screen = screen
        self.gridNumberX, self.gridNumberY = gridSize
        self.gridPixelSize = gridPixelSize
        self.backgroundColor = backgroundColor
        self.gridColor = gridColor
        self.gridLineWidth = gridLineWidth

    def __call__(self):
        upperBoundX = self.gridPixelSize * self.gridNumberX
        upperBoundY = self.gridPixelSize * self.gridNumberY
        self.screen.fill(self.backgroundColor)

        for gridIndexX in range(self.gridNumberX + 1):
            gridX = int(gridIndexX * self.gridPixelSize)
            pygame.draw.line(self.screen, self.gridColor, (gridX, 0), (gridX, upperBoundY), self.gridLineWidth)

        for gridIndexY in range(self.gridNumberY + 1):
            gridY = int(gridIndexY * self.gridPixelSize)
            pygame.draw.line(self.screen, self.gridColor, (0, gridY), (upperBoundX, gridY), self.gridLineWidth)



class DrawPointsAndSaveImage:
    def __init__(self, screen, drawGrid, drawReward, drawCircles, drawTrajectoryColor, drawPolicyArrows, gridPixelSize, fps = 60):
        self.screen = screen
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
            self.drawGrid()
            self.drawReward(rewardCoords)
            self.drawTrajectoryColor(trajectoryToDraw)
            self.drawPolicyArrows(trajectoryToDraw, trajectory, policy)
            pointsLocation = [list(np.array(agentState) * self.gridPixelSize - self.gridPixelSize//2)
                                  for agentState in state]
            self.drawCircles(pointsLocation)
            if saveImage:
                pygame.image.save(self.screen, saveImageDir + '/' + format(timeStep, '04') + ".png")


class DrawRewards:
    def __init__(self, screen, gridPixelSize, rewardIconPath):
        self.screen = screen
        self.gridPixelSize = gridPixelSize
        self.rewardIconPath = rewardIconPath

    def __call__(self, rewardCoords):
        rewardSymbol = pygame.image.load(self.rewardIconPath)
        rewardSymbol = pygame.transform.scale(rewardSymbol, (self.gridPixelSize//2, self.gridPixelSize//2))
        rewardsLocation = [np.array(coord) * self.gridPixelSize + self.gridPixelSize//4 for coord in rewardCoords]
        for rewardCoord in rewardsLocation:
            self.screen.blit(rewardSymbol, rewardCoord)



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
    def __init__(self, screen, leftTransformCoord, boxColors, gridSize):
        self.screen = screen
        self.leftTransformCoord = leftTransformCoord
        self.agentsColors = boxColors
        self.gridSize = gridSize

    def __call__(self, trajectoryToDraw):
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
                self.screen.blit(fillSquare, coord)



class DrawPolicyArrows:
    def __init__(self, screen, centerTransformCoord, arrowUnitSize, actionSpace, lineWidth = 3, lineCol = (0,0,0)):
        self.screen = screen
        self.centerTransformCoord = centerTransformCoord
        self.arrowUnitSize = arrowUnitSize
        self.actionSpace = actionSpace
        self.lineWidth = lineWidth
        self.lineCol = lineCol

    def __call__(self, trajectoryToDraw, trajectory, policy):
        timeSpan = len(trajectory)-1
        agentNumber = len(trajectory[0])
        for time in range(timeSpan):
            currentState = trajectory[time]
            oneStateDist = policy[currentState]
            for agentID in range(agentNumber):
                arrowCenter = self.centerTransformCoord(np.array(currentState[agentID]) + 1)

                for action in self.actionSpace:
                    prob = np.sum([oneStateDist[actionPair] for actionPair in oneStateDist.keys() if actionPair[agentID] == action])
                    arrowLength = prob * self.arrowUnitSize
                    if action !=(0,0):
                        direction = np.array(action)/np.linalg.norm(action, ord=2)
                        endPos = (arrowCenter + arrowLength* direction).astype(int)
                        pygame.draw.line(self.screen, self.lineCol, arrowCenter, endPos, self.lineWidth)
                        pygame.draw.circle(self.screen, self.lineCol, endPos, int(2 * self.lineWidth))
                    else:
                        pygame.draw.circle(self.screen, self.lineCol, arrowCenter, int(2 * self.lineWidth))


class DrawShade:
    def __init__(self, screen, transformCoord, gridSize):
        self.screen = screen
        self.transformCoord = transformCoord
        self.gridSize = gridSize

    def __call__(self, shadeUpperLeftCoord, originalColor, shadeGridWidth = 1, shadeGridHeight = 1, shadeAlpha = 50):
        shadeColor = originalColor + (shadeAlpha,)
        shadeWidthPix = shadeGridWidth* self.gridSize
        shadeHeightPix = shadeGridHeight* self.gridSize
        fillSquare = pygame.Surface((shadeWidthPix, shadeHeightPix), pygame.SRCALPHA)
        pygame.draw.rect(fillSquare, shadeColor, fillSquare.get_rect())
        shadePos = self.transformCoord(shadeUpperLeftCoord)
        self.screen.blit(fillSquare, shadePos)


















