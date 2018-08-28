import sys
import numpy

from .base_agent import BaseAgent
from human_modeling_utils import utils
from constant import *

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing


kEps = 0.02

kTunedParameters = [{
    'kernel': ['rbf'],
    'gamma': [2**e for e in [-6, -2, 2, 6]],
    'C': [2**e for e in [-6, -2, 2, 6]],
}]

kFold = 5

class ContextualBanditSVMProbAgent(BaseAgent):
    """
    This agent exploits contextual bandit algorithms. It is inspired by linUCB algorithm. The
    agent keeps two bandits: not-sending notification bandit, and sending bandit. We don't
    explicitly model not-sending bandit because the reward is always 0. For the sending bandit,
    instead of using ridge regression to estimate the reward from the state, we SVM here because
    the outcomes are only a positive and a negative reward.

    The state is represented by 5-tuple. Since all the elements are discrete, we use one-hot
    encoding to represent the state and feed it to the SVM classifier.
    """

    def getAction(self, stateTime, stateDay, stateLocation, stateActivity, stateLastNotification):
        super().getAction(stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)
        self.currentX = self.encode(stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)

        # the result should be based on the expected reward of two bandits (i.e., non-sending and
        # sending bandits) and choose whichever gives larger reward. Since non-sending bandit
        # always gives 0 reward, what matters is whether the reward from sending bandit is positive
        # or negative. On top of that, we encourage sending bandit to explore with a certain
        # probability regardless the outcome.
        if numpy.random.uniform() < kEps:
            self.chosenAction = True
        else:
            self.chosenAction = self.getTestLabel(self.currentX)

        return self.chosenAction
    
    def feedReward(self, reward):
        super().feedReward(reward)

        if self.chosenAction:
            self.xData.append(self.currentX)
            self.yData.append(1 if reward > 0 else 0)
            self.scaler = preprocessing.StandardScaler().fit(numpy.array(self.xData))
            xScaledData = self.scaler.transform(self.xData)
            
            if self.countDown <= 0:
                # try to train the model if the amount of data is sufficient
                try:
                    svc = SVC(probability=True)
                    self.clf = GridSearchCV(svc, kTunedParameters, cv=kFold, n_jobs=8)
                    yDataNumpyFormat = numpy.array(self.yData)
                    self.clf.fit(xScaledData, yDataNumpyFormat)
                    self.countDown = max(1, int(len(self.xData) ** 0.5))
                except:
                    import traceback
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    print(exc_type, exc_value, exc_traceback)
                    self.clf = None
            self.countDown -= 1
    
    def generateInitialModel(self):
        # the history of sending-notification bandit
        self.xData = []
        self.yData = []  # 1: positive reward, 0: negative reward
        self.clf = None
        self.countDown = 0
    
    def loadModel(self, filepath):
        sys.stderr.write("Warning: loadModel() does not support\n")        
    
    def saveModel(self, filepath):
        sys.stderr.write("Warning: saveModel() does not support\n")

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
    
    def getTestLabel(self, testX):
        try:
            scaledTestX = self.scaler.transform(numpy.array([testX]))
            res = self.clf.predict_proba(scaledTestX)
            res = res[0]  # get the first row and the only row
            print(res)
            expectedReward = res[0] * self._getNegativeReward() + res[1] * 1.
            return (expectedReward > 0.)
        except:
            # if we don't have enough data (insufficient history), we cannot make a prediction.
            # The best we can do is to encourage exploration
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_type, exc_value, exc_traceback)
            return True
