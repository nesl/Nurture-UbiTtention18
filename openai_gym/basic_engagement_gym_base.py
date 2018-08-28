"""Example of a custom gym environment. Run this for a demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import os

import gym
from gym.spaces import Discrete, Tuple
from gym.envs.registration import EnvSpec

from constant import *
from environment import *
from behavior import *
from human_modeling_utils import utils
from human_modeling_utils.chronometer import Chronometer


class BasicEngagementGymBase(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config=None):

        if config is None:
            config = self._get_default_config()

        self.rewardCriteria = config['rewardCriteria']
        self.environment = config['environment']
        self.behavior = config['behavior']
        self.verbose = config['verbose']
        self.episodeLengthDay = config['episodeLengthDay']
        self.stepSizeMinute = config['stepSizeMinute']

        self.action_space = Discrete(2)
        self.observation_space = self.get_observation_space()
        self._spec = EnvSpec("EngagementGym-v0")

        self.masterNumDayPassed = 0


    @abc.abstractmethod
    def get_observation_space(self):
        """
        Define observation space
        """

    @abc.abstractmethod
    def intepret_state(self, state_tuple):
        """
        Interface the default state format to the agent understanable format
        """

    def reset(self):
        # set chronometer which automatically skips 10pm to 8am because it's usually when people
        # sleep
        self.chronometer = Chronometer(
                refDay=self.masterNumDayPassed % 7,
                skipFunc=(lambda hour, _m, _d: hour < 8 or hour >= 22),
        )

        self.lastNotificationMinute = 0
        self.lastNotificationHour = 0
        self.lastNotificationNumDays = 0

        # statistics
        self.simulationResults = []
        self.totalReward = 0.
        self.numSteps = 0
        
        # fast forward a little bit to the next available time
        numDaysPassed, currentHour, currentMinute, currentDay = self.chronometer.forward(
                self.stepSizeMinute)

        return self.intepret_state(self._generate_state())

    def step(self, action):

        assert action in [0, 1]  # 0: silent, 1: send notification

        reward = self._generate_reward(action)
        
        # prepare for the next state
        numDaysPassed, currentHour, currentMinute, currentDay = self.chronometer.forward(
                self.stepSizeMinute)
        if self.verbose:
            print("Day %d %d:%02d" % (numDaysPassed, currentHour, currentMinute))

        gymState = self.intepret_state(self._generate_state())

        self.totalReward += reward
        self.numSteps += 1

        done = (numDaysPassed > self.episodeLengthDay)
        if done:
            self.masterNumDayPassed += self.episodeLengthDay
            
            # print some intermediate results
            print()
            print("===== end of episode, %d days passed in total ====" % self.masterNumDayPassed)
            self._printResults(self.simulationResults)

        return gymState, reward, done, {}

    def _generate_state(self):

        # retrieve current state
        numDaysPassed, currentHour, currentMinute, currentDay = self.chronometer.getCurrentTime()

        # get environment info (user context)
        self.lastNotificationTime = utils.getDeltaMinutes(
                numDaysPassed, currentHour, currentMinute,
                self.lastNotificationNumDays, self.lastNotificationHour, self.lastNotificationMinute,
        )
        self.stateLastNotification = utils.getLastNotificationState(self.lastNotificationTime)
        self.stateLocation, self.stateActivity = self.behavior.getLocationActivity(
                currentHour, currentMinute, currentDay)

        # prepare observables and get action
        self.stateTime = utils.getTimeState(currentHour, currentMinute)
        self.stateDay = utils.getDayState(currentDay)

        return (
                self.stateTime,
                self.stateDay,
                self.stateLocation,
                self.stateActivity,
                self.stateLastNotification,
        )

    def _generate_reward(self, action):
        
        # retrieve current state
        numDaysPassed, currentHour, currentMinute, currentDay = self.chronometer.getCurrentTime()

        # get probability of each possible user reaction
        probReactions = self.environment.getResponseDistribution(
                currentHour, currentMinute, currentDay,
                self.stateLocation, self.stateActivity, self.lastNotificationTime,
        )
        probReactions = utils.normalize(*probReactions)
        probAnsweringNotification, probIgnoringNotification, probDismissingNotification = probReactions
        
        # calculate reward
        sendNotification = (action == 1)
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

        # log this session
        self.simulationResults.append({
                'context': {
                    'numDaysPassed': numDaysPassed,
                    'hour': currentHour,
                    'minute': currentMinute,
                    'day': currentDay,
                    'location': self.stateLocation,
                    'activity': self.stateActivity,
                    'lastNotification': self.lastNotificationTime,
                },
                'probOfAnswering': probAnsweringNotification,
                'probOfIgnoring': probIgnoringNotification,
                'probOfDismissing': probDismissingNotification,
                'decision': sendNotification,
                'reward': reward,
        })

        return reward

    def _printResults(self, results):
        notificationEvents = [r for r in results if r['decision']]
        numNotifications = len(notificationEvents)
        numAcceptedNotis = len([r for r in notificationEvents if r['reward'] > 0])
        numDismissedNotis = len([r for r in notificationEvents if r['reward'] < 0])
        
        answerRate = numAcceptedNotis / numNotifications if numNotifications > 0 else 0.
        dismissRate = numDismissedNotis / numNotifications if numNotifications > 0 else 0.
        numActionedNotis = numAcceptedNotis + numDismissedNotis
        responseRate = numAcceptedNotis / numActionedNotis if numActionedNotis > 0 else 0.

        totalReward = sum([r['reward'] for r in results])

        expectedNumDeliveredNotifications = sum([r['probOfAnswering'] for r in results])
        deltaDays = results[-1]['context']['numDaysPassed'] - results[0]['context']['numDaysPassed'] + 1

        print("  reward=%f / step=%d (%f)" % (totalReward, len(results), totalReward / len(results)))
        print("  %d notifications have been sent (%.1f / day):" % (numNotifications, numNotifications / deltaDays))
        print("    - %d are answered (%.2f%%)"  % (numAcceptedNotis, answerRate * 100.))
        print("    - %d are dismissed (%.2f%%)"  % (numDismissedNotis, dismissRate * 100.))
        print("    - response rate: %.2f%%"  % (responseRate * 100.))
        print("  Expectation of total delivered notifications is %.2f" % expectedNumDeliveredNotifications)

    def _filterByWeek(self, results, week):
        startDay = week * 7
        endDay = startDay + 7
        return [r for r in results
                if startDay <= r['context']['numDaysPassed'] < endDay]
    
    def _get_default_config(self):
   
        ### simulation configuration
        rewardCriteria = {
                ANSWER_NOTIFICATION_ACCEPT: 1,
                ANSWER_NOTIFICATION_IGNORE: 0,
                ANSWER_NOTIFICATION_DISMISS: -5,
        }

        verbose = False

        ### environment

        project_root_folder = os.path.join(os.path.dirname(__file__), '..')

        #environment = AlwaysSayOKUser()
        #environment = StubbornUser()
        #environment = LessStubbornUser()
        #environment = SurveyUser('survey/ver1_pilot/data/02.txt')

        mturk_data_folder = os.path.join(project_root_folder, 'survey/ver2_mturk/results/')
        environment = MTurkSurveyUser(
                filePaths=[
                    os.path.join(mturk_data_folder, '01_1st_Batch_3137574_batch_results.csv'),
                    os.path.join(mturk_data_folder, '02_Batch_3148398_batch_results.csv'),
                    os.path.join(mturk_data_folder, '03_Batch_3149214_batch_results.csv'),
                ],
                filterFunc=(lambda r: ord(r['rawWorkerID'][-1]) % 3 == 2),
        )

        ### user daily routing modevior = RandomBehavior()

        behavior_data_folder = os.path.join(project_root_folder, 'behavior/data/')
        #behavior = ExtraSensoryBehavior(os.path.join(behavior_data_folder, '2.txt'))
        #behavior = ExtraSensoryBehavior(os.path.join(behavior_data_folder, '4.txt'))
        #behavior = ExtraSensoryBehavior(os.path.join(behavior_data_folder, '5.txt'))
        behavior = ExtraSensoryBehavior(os.path.join(behavior_data_folder, '6.txt'))

        episodeLengthDay = 1
        stepSizeMinute = 1

        return {
                "rewardCriteria": rewardCriteria,
                "environment": environment,
                "behavior": behavior,
                "verbose": verbose,
                "episodeLengthDay": episodeLengthDay,
                "stepSizeMinute": stepSizeMinute,
        }

