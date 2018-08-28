import sys

from .base_agent import BaseAgent

class AlwaysSendNotificationAgent(BaseAgent):
    
    def getAction(self, stateTime, stateDay, stateLocation, stateActivity, stateLastNotification):
        super().getAction(stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)
        return True
    
    def feedReward(self, reward):
        super().feedReward(reward)
        # who cares about the reward
    
    def generateInitialModel(self):
        pass
    
    def loadModel(self, filepath):
        sys.stderr.write("Warning: loadModel() does not support\n")        
    
    def saveModel(self, filepath):
        sys.stderr.write("Warning: saveModel() does not support\n")        
