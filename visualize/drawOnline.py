import pygame
import numpy as np
import sys


class TransformCoord:
    def __init__(self, gridSize, location, edgeSize = 3):
        self.gridSize = gridSize
        self.edgeSize = edgeSize
        self.location = location

    def __call__(self, coords):
        if self.location == 'center':  # make the coords the center of the shape
            xLoc = int((coords[0] - 1) * self.gridSize + self.edgeSize + self.gridSize / 2)
            yLoc = int((coords[1] - 1) * self.gridSize + self.edgeSize + self.gridSize / 2)
        else:
            xLoc = int((coords[0] - 1) * self.gridSize + self.edgeSize)
            yLoc = int((coords[1] - 1) * self.gridSize + self.edgeSize)

        return (xLoc, yLoc)

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


class DrawCircles:
    def __init__(self, screen, centerTransformCoord, colorList, pointsWidth):
        self.screen = screen
        self.centerTransformCoord = centerTransformCoord
        self.pointsWidth = pointsWidth
        self.colorList = colorList

    def __call__(self, state):
        for agentIndex in range(len(self.colorList)):
            agentState = state[agentIndex]
            agentLocation = self.centerTransformCoord(np.array(agentState) + 1)
            agentColor = self.colorList[agentIndex]
            agentWidth = self.pointsWidth[agentIndex]
            pygame.draw.circle(self.screen, agentColor, agentLocation, agentWidth)


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

class DrawTrajectoryColor:
    def __init__(self, drawShade, agentsColor):
        self.drawShade = drawShade
        self.agentsColor = agentsColor

    def __call__(self, state):
        agentNumber = len(state)
        for agentIndex in range(agentNumber):
            agentPosition = np.array(state[agentIndex]) + 1
            agentColor = self.agentsColor[agentIndex]
            self.drawShade(agentPosition, agentColor)


class DrawPolicyArrows:
    def __init__(self, screen, centerTransformCoord, arrowUnitSize, actionSpace, lineWidth = 3, lineCol = (0,0,0)):
        self.screen = screen
        self.centerTransformCoord = centerTransformCoord
        self.arrowUnitSize = arrowUnitSize
        self.actionSpace = actionSpace
        self.lineWidth = lineWidth
        self.lineCol = lineCol

    def __call__(self, state, policy):
        agentNumber = len(state)
        actionDist = policy[state]
        pointsLocation = np.array(state) + 1

        for agentID in range(agentNumber):
            agentLocation = pointsLocation[agentID]
            arrowCenter = self.centerTransformCoord(agentLocation)

            for action in self.actionSpace:
                prob = np.sum([actionDist[actionPair] for actionPair in actionDist.keys() if actionPair[agentID] == action])
                arrowLength = prob * self.arrowUnitSize
                if action !=(0,0):
                    direction = np.array(action)/np.linalg.norm(action, ord=2)
                    endPos = (arrowCenter + arrowLength* direction).astype(int)
                    pygame.draw.line(self.screen, self.lineCol, arrowCenter, endPos, self.lineWidth)
                    pygame.draw.circle(self.screen, self.lineCol, endPos, int(2 * self.lineWidth))
                else:
                    pygame.draw.circle(self.screen, self.lineCol, arrowCenter, int(2 * self.lineWidth))


class Display:
    def __init__(self, screen, drawGrid, drawRewards, drawCircles, drawTrajectoryColor, drawPolicyArrows,
                 pointExtendTime = 100, FPS = 60):
        self.screen = screen
        self.FPS = FPS
        self.pointExtendFrame = int(pointExtendTime * self.FPS / 1000)

        self.drawGrid = drawGrid
        self.drawRewards = drawRewards
        self.drawCircles = drawCircles
        self.drawTrajectoryColor = drawTrajectoryColor
        self.drawPolicyArrows = drawPolicyArrows

    def __call__(self, rewardCoords, trajectory, policy, saveImageDir = None, saveImage = False):
        fpsClock = pygame.time.Clock()

        for timeStep in range(len(trajectory)):
            fpsClock.tick(self.FPS)
            self.drawGrid()
            self.drawRewards(rewardCoords)

            passedStates = trajectory[:timeStep+1]
            for previousState in passedStates:
                self.drawTrajectoryColor(previousState)
                self.drawPolicyArrows(previousState, policy)

            state = trajectory[timeStep]
            for frameNumber in range(self.pointExtendFrame):
                self.drawCircles(state)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                pygame.display.update()

            if saveImage:
                pygame.image.save(self.screen, saveImageDir + '/' + format(timeStep, '04') + ".png")
