import random
import numpy as np

from constant import *
from human_modeling_utils import utils
from human_modeling_utils.chronometer import Chronometer


class Controller:

    def __init__(self, agent, environment, behavior,
            simulationWeek=10, negativeReward=-10, verbose=True):
        self.rewardCriteria = {
                ANSWER_NOTIFICATION_ACCEPT: 1,
                ANSWER_NOTIFICATION_IGNORE: 0,
                ANSWER_NOTIFICATION_DISMISS: negativeReward,
        }

        self.verbose = verbose

        # set chronometer which automatically skips 10pm to 8am because it's usually when people
        # sleep
        self.chronometer = Chronometer(skipFunc=(lambda hour, _m, _d: hour < 8 or hour >= 22))

        self.stepWidthMinutes = 10
        self.simulationWeek = simulationWeek

        self.lastNotificationMinute = 0
        self.lastNotificationHour = 0
        self.lastNotificationNumDays = 0

        self.agent = agent
        self.environment = environment
        self.behavior = behavior

        self.simulationResults = []

        self.agent.setNegativeReward(negativeReward)

    def execute(self):
        numDaysPassed, currentHour, currentMinute, currentDay = self.chronometer.forward(
                self.stepWidthMinutes)

        while numDaysPassed < self.simulationWeek * 7:
            if self.verbose:
                print("Day %d %d:%02d" % (numDaysPassed, currentHour, currentMinute))

            # get environment info (user context)
            lastNotificationTime = utils.getDeltaMinutes(
                    numDaysPassed, currentHour, currentMinute,
                    self.lastNotificationNumDays, self.lastNotificationHour, self.lastNotificationMinute,
            )
            stateLastNotification = utils.getLastNotificationState(lastNotificationTime)
            stateLocation, stateActivity = self.behavior.getLocationActivity(
                    currentHour, currentMinute, currentDay)
            probAnsweringNotification, probIgnoringNotification, probDismissingNotification = (
                    self.environment.getResponseDistribution(
                        currentHour, currentMinute, currentDay,
                        stateLocation, stateActivity, lastNotificationTime,
                    )
            )
            probAnsweringNotification, probIgnoringNotification, probDismissingNotification = utils.normalize(
                    probAnsweringNotification, probIgnoringNotification, probDismissingNotification)

            # prepare observables and get action
            stateTime = utils.getTimeState(currentHour, currentMinute)
            stateDay = utils.getDayState(currentDay)
            sendNotification = self.agent.getAction(stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)

            # calculate reward
            if not sendNotification:
                reward = 0
            else:
                userReaction = np.random.choice(
                        a=[ANSWER_NOTIFICATION_ACCEPT, ANSWER_NOTIFICATION_IGNORE, ANSWER_NOTIFICATION_DISMISS],
                        p=[probAnsweringNotification, probIgnoringNotification, probDismissingNotification],
                )
                reward = self.rewardCriteria[userReaction]
                self.lastNotificationNumDays = numDaysPassed
                self.lastNotificationHour = currentHour
                self.lastNotificationMinute = currentMinute
            self.agent.feedReward(reward)

            # log this session
            self.simulationResults.append({
                    'context': {
                        'numDaysPassed': numDaysPassed,
                        'hour': currentHour,
                        'minute': currentMinute,
                        'day': currentDay,
                        'location': stateLocation,
                        'activity': stateActivity,
                        'lastNotification': lastNotificationTime,
                    },
                    'probOfAnswering': probAnsweringNotification,
                    'probOfIgnoring': probIgnoringNotification,
                    'probOfDismissing': probDismissingNotification,
                    'decision': sendNotification,
                    'reward': reward,
            })

            # get the next decision time point
            numDaysPassed, currentHour, currentMinute, currentDay = self.chronometer.forward(
                    self.stepWidthMinutes)

        return self.simulationResults
