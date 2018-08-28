import sys
import numpy as np

from .base_agent import BaseAgent
from constant import *


class BaseClassifierAgent(BaseAgent):

    def __init__(self, negRewardWeight=3):
        super().__init__()
        self.negRewardWeight = negRewardWeight

    def loadModel(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
        mat = np.array([list(map(int, l.strip().split(','))) for l in lines])

        trainX = []
        trainY = []
        for reward, t, d, l, a, ln in mat:
            vec = self.encode(t, d, l, a, ln)
            repeat = 1 if reward >= 0 else self.negRewardWeight
            for _ in range(repeat):
                trainX.append(vec)
                trainY.append(reward)

        trainX = np.array(trainX)
        trainY = np.array(trainY)

        self.model = self.trainModel(trainX, trainY)
    
    def getAction(self, stateTime, stateDay, stateLocation, stateActivity, stateLastNotification):
        super().getAction(stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)
        self.currentX = self.encode(stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)
        self.expectedReward = self.getRewardLabel(self.model, self.currentX)
        return self.expectedReward > 0
    
    def encode(self, stateTime, stateDay, stateLocation, stateActivity, stateLastNotification):
        return [
            1 if stateTime == STATE_TIME_MORNING else 0,
            1 if stateTime == STATE_TIME_AFTERNOON else 0,
            1 if stateTime == STATE_TIME_EVENING else 0,
            1 if stateTime == STATE_TIME_SLEEPING else 0,
            1 if stateDay == STATE_DAY_WEEKDAY else 0,
            1 if stateDay == STATE_DAY_WEEKEND else 0,
            1 if stateLocation == STATE_LOCATION_HOME else 0,
            1 if stateLocation == STATE_LOCATION_WORK else 0,
            1 if stateLocation == STATE_LOCATION_OTHER else 0,
            1 if stateActivity == STATE_ACTIVITY_STATIONARY else 0,
            1 if stateActivity == STATE_ACTIVITY_WALKING else 0,
            1 if stateActivity == STATE_ACTIVITY_RUNNING else 0,
            1 if stateActivity == STATE_ACTIVITY_DRIVING else 0,
            1 if stateLastNotification == STATE_LAST_NOTIFICATION_WITHIN_1HR else 0,
            1 if stateLastNotification == STATE_LAST_NOTIFICATION_LONG else 0,
        ]

    def trainModel(self, dataX, dataY):
        sys.stderr.write("ERROR: Please implement trainModel()\n")
        exit(0)

    def getRewardLabel(self, model, dataXVec):
        sys.stderr.write("ERROR: Please implement getRewardLabel()\n")
        exit(0)
