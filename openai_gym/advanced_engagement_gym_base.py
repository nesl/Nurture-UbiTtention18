from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import os
import sys

import gym
from gym.spaces import Discrete, Tuple
from gym.envs.registration import EnvSpec

from constant import *
from environment import *
from behavior import *
from human_modeling_utils import utils
from human_modeling_utils.chronometer import Chronometer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Nurture', 'server', 'notification'))
from nurture.learning.state import State


class AdvancedEngagementGymBase(gym.Env):
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
        self.screenStatusProb = config['screenStatusProb']
        self.ringerModeProb = config['ringerModeProb']
        self.responseAdjustmentScreen = config['responseAdjustmentScreen']
        self.responseAdjustmentRinger = config['responseAdjustmentRinger']

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
    def intepret_state(self, state):
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
        self.stateLocation, self.stateActivity = self.behavior.getLocationActivity(
                currentHour, currentMinute, currentDay)

        self.state = State(
            timeOfDay=self._get_time_of_day(currentHour, currentMinute),
            dayOfWeek=self._get_day_of_week(currentDay, currentHour, currentMinute),
            motion=self._get_motion(self.stateActivity),
            location=self.stateLocation,
            notificationTimeElapsed=self.lastNotificationTime,
            ringerMode=self._pick_ringer_mode(),
            screenStatus=self._pick_screen_status(),
        )
        return self.state

    def _generate_reward(self, action):
        
        # retrieve current state
        numDaysPassed, currentHour, currentMinute, currentDay = self.chronometer.getCurrentTime()

        # get probability of each possible user reaction
        probReactions = self.environment.getResponseDistribution(
                currentHour, currentMinute, currentDay,
                self.stateLocation, self.stateActivity, self.lastNotificationTime,
        )
        probReactions = utils.normalize(*probReactions)
        probReactions = self._adjust_prob_by_screen_ringer(self.state, *probReactions)
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
    
    def _get_time_of_day(self, currentHour, currentMinute):
        return currentHour / 24. + currentMinute / 24. / 60.

    def _get_day_of_week(self, currentDay, currentHour, currentMinute):
        return currentDay / 7. + currentHour / 7. / 24. + currentMinute / 7. / 24. / 60.

    def _get_motion(self, original_activity):
        mapping = {
            STATE_ACTIVITY_STATIONARY: State.MOTION_STATIONARY,
            STATE_ACTIVITY_WALKING: State.MOTION_WALKING,
            STATE_ACTIVITY_RUNNING: State.MOTION_RUNNING,
            STATE_ACTIVITY_DRIVING: State.MOTION_DRIVING,
            STATE_ACTIVITY_COMMUTE: State.MOTION_DRIVING,
        }
        return mapping[original_activity] 

    def _pick_ringer_mode(self):
        return np.random.choice(
                a=list(self.ringerModeProb.keys()),
                p=list(self.ringerModeProb.values()),
        )

    def _pick_screen_status(self):
        return np.random.choice(
                a=list(self.screenStatusProb.keys()),
                p=list(self.screenStatusProb.values()),
        )

    def _adjust_prob_by_screen_ringer(self, state, p_answer, p_ignore, p_dismiss):
        p_answer = (p_answer + 0.01) * self.responseAdjustmentScreen[state.screenStatus]
        p_answer = (p_answer + 0.01) * self.responseAdjustmentRinger[state.ringerMode]
        return utils.normalize(p_answer, p_ignore, p_dismiss)

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
        deltaDays = results[-2]['context']['numDaysPassed'] - results[0]['context']['numDaysPassed'] + 1

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
                #filterFunc=(lambda r: ord(r['rawWorkerID'][-1]) % 3 == 2),
                filterFunc=(lambda _: True),
        )

        ### user daily routing modevior = RandomBehavior()

        behavior_data_folder = os.path.join(project_root_folder, 'behavior/data/')
        #behavior = ExtraSensoryBehavior(os.path.join(behavior_data_folder, '2.txt'))
        #behavior = ExtraSensoryBehavior(os.path.join(behavior_data_folder, '4.txt'))
        #behavior = ExtraSensoryBehavior(os.path.join(behavior_data_folder, '5.txt'))
        behavior = ExtraSensoryBehavior(os.path.join(behavior_data_folder, '6.txt'))

        episodeLengthDay = 1
        stepSizeMinute = 1
        
        screenStatusProb = {
                State.SCREEN_STATUS_ON: 0.15,
                State.SCREEN_STATUS_OFF: 0.85,
        }
        ringerModeProb = {
                State.RINGER_MODE_SILENT: 0.4,
                State.RINGER_MODE_VIBRATE: 0.3,
                State.RINGER_MODE_NORMAL: 0.3,
        }
        responseAdjustmentScreen = {
                State.SCREEN_STATUS_ON: 100.,
                State.SCREEN_STATUS_OFF: 1.,
        }
        responseAdjustmentRinger = {
                State.RINGER_MODE_SILENT: 1.,
                State.RINGER_MODE_VIBRATE: 1.,
                State.RINGER_MODE_NORMAL: 1.,
        }

        return {
                "rewardCriteria": rewardCriteria,
                "environment": environment,
                "behavior": behavior,
                "verbose": verbose,
                "episodeLengthDay": episodeLengthDay,
                "stepSizeMinute": stepSizeMinute,
                "screenStatusProb": screenStatusProb,
                "ringerModeProb": ringerModeProb,
                "responseAdjustmentScreen": responseAdjustmentScreen,
                "responseAdjustmentRinger": responseAdjustmentRinger,
        }

