from constant import *

from human_modeling_utils import utils


class BaseAgent:
    """
    There are two operating modes for an agent. The first is iteractive operating mode, which
    follows a standard state-action-reward sequence. It can be used in on-policy learning. The
    second mode is batch operating mode. It is designed for the agent which wants to introduce
    a notion of episodes.

    The following is the expected flow for the iterative mode:

        for each step:
            agent.getAction(*state)
            ...
            agent.feedReward(reward)

    The following is the expected flow for the batch mode:

        for each episode:
            # render a sequence of states
            for each step, we grab a state:
                agent.getAction(*state)
            
            # feed the reward history to the agent
            agent.feedBatchReward(history)
    """

    STAGE_WAIT_ACTION = 0
    STAGE_WAIT_REWARD = 1

    MODE_ITERATIVE = 0
    MODE_BATCH = 1

    def __init__(self, operatingMode=None):
        self.stage = BaseAgent.STAGE_WAIT_ACTION
        self.generateInitialModel()
        self.negativeReward = -5
        self.operatingMode = (operatingMode if operatingMode is not None
                else BaseAgent.MODE_ITERATIVE)

    def setNegativeReward(self, negativeReward):
        self.negativeReward = negativeReward

    def _getNegativeReward(self):
        return self.negativeReward

    def getAction(self, stateTime, stateDay, stateLocation, stateActivity, stateLastNotification):
        """
        The function feedObservation() receives the 4-tuple elements (i.e., time, location,
        activity, and time elapsed since last notification) and makes a decision of sending a
        notificatiion or not.

        The function is anticipated to be provided implementation

        Returns:
          A bool indicating whether to send the notification or not
        """

        if self.operatingMode == BaseAgent.MODE_ITERATIVE:
            # check stage
            if self.stage != BaseAgent.STAGE_WAIT_ACTION:
                raise Exception("It is not in the stage of determining action")
            self.stage = BaseAgent.STAGE_WAIT_REWARD

        # check argument value 
        if stateTime not in utils.allTimeStates():
            raise Exception("Invalid stateTime value (got %d)" % stateTime)
        if stateDay not in utils.allDayStates():
            raise Exception("Invalid stateDay value (got %d)" % stateDay)
        if stateLocation not in utils.allLocationStates():
            raise Exception("Invalid stateLocation value (got %d)" % stateLocation)
        if stateActivity not in utils.allActivityStates():
            raise Exception("Invalid stateActivity value (got %d)" % stateActivity)
        if stateLastNotification not in utils.allLastNotificationStates():
            raise Exception("Invalid stateActivity value (got %d)" % stateLastNotification)

    def feedReward(self, reward):
        """
        After the agent gives out the action by the function `getAction()`, the controller is
        anticipated to signal the reward to this agent via this function `feedReward()`.
        
        The function is anticipated to be provided implementation
        """
        if self.operatingMode != BaseAgent.MODE_ITERATIVE:
            raise Exception("feedReward() is expected to call in the interative mode")

        if self.stage != BaseAgent.STAGE_WAIT_REWARD:
            raise Exception("It is not in the stage of receiving reward")
        self.stage = BaseAgent.STAGE_WAIT_ACTION
    
    def feedBatchRewards(self, history):
        """
        After querying a couple of `getAction()` calls, the agent then get the reward results
        in a batch in this function. 
        
        The function is anticipated to be provided implementation

        Params:
          - history: A list of (state, action, reward) tuples
            - state is a 5-tuple of (sTime, sDay, sLocation, sActivity, sLastNotificationTime)
        """
        if self.operatingMode != BaseAgent.MODE_BATCH:
            raise Exception("feedBatchRewards() is expected to call in the batch mode")

    def generateInitialModel(self):
        """
        To initialize the blank policy.
        """
        pass
    
    def loadModel(self, filepath):
        """
        The function loadModel() loads the predefined policy.
        """
        pass

    def saveModel(self, filepath):
        """
        The function saveModel() saves the current policy.
        """
        pass

