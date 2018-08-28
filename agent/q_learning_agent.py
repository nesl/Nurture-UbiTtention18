import sys
import numpy

from .base_agent import BaseAgent
from human_modeling_utils import utils
from constant import *

kInitialLearningRate = 1.0
kMinLearningRate = 0.003
kGamma = 1.0
kInitEps = 0.5
kMinEps = 0.1

class QLearningAgent(BaseAgent):

    def getAction(self, stateTime, stateDay, stateLocation, stateActivity, stateLastNotification):
        super().getAction(stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)
        state = (stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)
        self.currentState = state

        eps = max(kMinEps, kInitEps * (0.85 ** (self.numSteps // 100)))
        if numpy.random.random() < eps:
            self.chosenAction = numpy.random.choice([a for a in self.qTable[state]])
        else:
            self.chosenAction = utils.argmaxDict(self.qTable[state])
        return self.chosenAction
    
    def feedReward(self, reward):
        super().feedReward(reward)

        curStt, curAct = self.currentState, self.chosenAction
        curT, curD, curL, curA, curLN = curStt
        nxtLN = (STATE_LAST_NOTIFICATION_WITHIN_1HR if self.chosenAction else curLN)
        nxtStt = (curT, curD, curL, curA, nxtLN)

        eta = max(kMinLearningRate, kInitialLearningRate * (0.85 ** (self.numSteps // 100)))
        maxNextQVal = utils.maxDictVal(self.qTable[curStt])
        self.qTable[curStt][curAct] = self.qTable[curStt][curAct] + eta * (reward + kGamma * maxNextQVal - self.qTable[curStt][curAct])

        self.numSteps += 1
    
    def generateInitialModel(self):
        self.qTable = {}
        for sTime in utils.allTimeStates():
            for sDay in utils.allDayStates():
                for sLocation in utils.allLocationStates():
                    for sActivity in utils.allActivityStates():
                        for sNotification in utils.allLastNotificationStates():
                            state = (sTime, sDay, sLocation, sActivity, sNotification)
                            self.qTable[state] = {a: 0.0 for a in [True, False]}
        self.numSteps = 0
    
    def loadModel(self, filepath):
        sys.stderr.write("Warning: loadModel() does not support\n")        
    
    def saveModel(self, filepath):
        sys.stderr.write("Warning: saveModel() does not support\n")
