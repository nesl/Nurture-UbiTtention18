import sys
import numpy as np

from .base_agent import BaseAgent
from human_modeling_utils import utils
from constant import *


kProbeRate = 0.2

class ClassifierDataCollectionAgent(BaseAgent):

    def getAction(self, stateTime, stateDay, stateLocation, stateActivity, stateLastNotification):
        super().getAction(stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)
        state = (stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)
        self.currentState = state
        self.chosenAction = np.random.random() < kProbeRate
        return self.chosenAction
    
    def feedReward(self, reward):
        super().feedReward(reward)
        if self.chosenAction:
            curT, curD, curL, curA, curLN = self.currentState
            self.data.append([reward, curT, curD, curL, curA, curLN])
    
    def generateInitialModel(self):
        self.data = []  # a list of lists, each child list has
                        # [reward, stateTime, stateDay, stateLoc, stateAct, stateNotiTime]
    
    def loadModel(self, filepath):
        sys.stderr.write("Warning: loadModel() does not support\n")        
    
    def saveModel(self, filepath):
        with open(filepath, 'w') as fo:
            for data in self.data:
                fo.write("%s\n" % ",".join(list(map(str, data))))
