import numpy as np

from .base_behavior_model import BaseBehaviorModel
from constant import *


class RandomBehavior(BaseBehaviorModel):
    
    def getLocationActivity(self, hour, minute, day):
        stateLocation = np.random.choice(
            a=[STATE_LOCATION_HOME, STATE_LOCATION_WORK, STATE_LOCATION_OTHER],
            p=[0.5, 0.4, 0.1],
        )
        stateActivity = np.random.choice(
            a=[STATE_ACTIVITY_STATIONARY, STATE_ACTIVITY_WALKING, STATE_ACTIVITY_RUNNING, STATE_ACTIVITY_DRIVING],
            p=[0.7, 0.1, 0.1, 0.1],
        )
        return (stateLocation, stateActivity)
