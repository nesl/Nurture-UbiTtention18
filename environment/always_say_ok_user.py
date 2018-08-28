import numpy

from constant import *
from .base_environment import BaseEnvironment

class AlwaysSayOKUser(BaseEnvironment):

    def getResponseDistribution(self, hour, minute, day,
            stateLocation, stateActivity, lastNotificationTime):
        return (
                1.0,  # probAnsweringNotification
                0.0,  # probIgnoringNotification
                0.0,  # probDismissingNotification
        )
