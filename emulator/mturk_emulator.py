import os
import dill
import shutil
import csv

from utils import utils
from constant import *
from utils.chronometer import Chronometer
from survey.ver3_mturk_interactive.mturk_survey_generator import MTurkSurveyGenerator
from .utils import *


class MTurkEmulator:
    """
    `MTurkEmulator` emulates the process of notification-response cycle. This emulator treats the
    crowd on mTurk as the notification system consumer and send notifications to her. The emulator
    emulates several rounds. The first 7 rounds will be a length of one single day. After that,
    each round lasts for a week.

    The entire emulation process includes multiple round. Each round has two phases, generating
    batch notifications and receiving reward results. Typically, after the notifications are
    generated, the questions have to be uploaded to mTurk, hence, the process has to be paused
    and saved. Once we collect the results from mTurk, we can resume the process and finish the
    phase of receiving reward results. Thus, the execution cycle should look like the following:

        [resume, program starts]
          - execute receiving reward result phase of round n
          - execute generating notification phase of round n+1
          - generate a save point
        [pause, program stops]

    The context of a emulator will be saved in a folder, including generated notification files,
    response results from mTurk, and save points. The formats of the file names are:

        - mTurk survey file (notifications):   ddd-ddd.action.*.csv    (e.g., 007-013.action.csv)
        - emulator save point:                 ddd-ddd.savepoint.p     (e.g., 007-013.savepoint.p)
        - mTurk results (responses):           ddd-ddd.response.*.csv  (e.g., 007-013.response.Batch_3149214_batch_results.csv)

    The prefix ddd-ddd present the start day and the end day of that round.
    """

    ACTION_GENERATING_BATCH_NOTIFICATION = 0
    ACTION_RECEIVING_REWARDS = 1

    @staticmethod
    def createEmulator(folderPath, behavior, agent, overWrite=False, negativeReward=-5,
            verbose=True):
        if os.path.exists(folderPath):
            if overWrite:
                shutil.rmtree(folderPath)
            else:
                raise Exception("Folder \"%s\" has been existed" % folderPath)

        os.mkdir(folderPath)
        return MTurkEmulator(folderPath, behavior, agent, negativeReward, verbose)

    @staticmethod
    def restoreEmulator(folderPath):
        # find the last save point
        lastSavePointFile = None
        lastStartDay, lastEndDay = None, None
        curStartDay = 0
        while True:
            endDay = MTurkEmulator.getRoundEndDay(curStartDay)
            savePointFile = MTurkEmulator.getFile(folderPath, curStartDay, endDay,
                    fileType='savepoint', extension='p')
            if savePointFile is None:
                break
            lastSavePointFile = savePointFile
            lastStartDay, lastEndDay = curStartDay, endDay
            curStartDay = endDay + 1
        
        if lastSavePointFile is None:
            raise Exception("No save point file can be found")

        # integrity check
        curStartDay = 0
        while curStartDay < lastStartDay:
            endDay = MTurkEmulator.getRoundEndDay(curStartDay)
            
            for fileType in ['action', 'response']:
                tmpFile = MTurkEmulator.getFile(folderPath, curStartDay, endDay,
                        fileType, extension='csv')
                if tmpFile is None:
                    raise Exception("Integrity check failed: No %s file during %03d-%03d."
                            % (fileType, curStartDay, endDay))

            curStartDay = endDay + 1

        tmpFile = MTurkEmulator.getFile(folderPath, lastStartDay, lastEndDay,
                fileType='action', extension='csv')
        if tmpFile is None:
            raise Exception("Integrity check failed: No action file during %03d-%03d."
                    % (lastStartDay, lastEndDay))

        lastSavePointPath = os.path.join(folderPath, lastSavePointFile)
        emulator = dill.load(open(lastSavePointPath, "rb"))
        emulator._updateFolderPath(folderPath)
        return emulator

    @staticmethod
    def getRoundEndDay(startDay):
        # The rounds in the first week last only one single day. The rest of founds last a week
        # long.
        if startDay < 7:
            return startDay
        return startDay + 6

    @staticmethod
    def getFile(folderPath, startDay, endDay, fileType, extension, raiseException=True):
        """
        Returns the file name with the given constraints, or `None` if no file is found.

        If `raiseException` flag is `True` and there are more than one matches, this function
        will throw an exception.
        """
        filePrefix = "%03d-%03d.%s." % (startDay, endDay, fileType)
        filePostfix = ".%s" % extension
        allFiles = [name for name in os.listdir(folderPath)
                if name.startswith(filePrefix) and name.endswith(filePostfix)]
        if len(allFiles) >= 2 and raiseException:
            raise Exception("More than one files have prefix \"%s\" and postfix \"%s\"."
                    % (filePrefix, filePostfix))
        return allFiles[0] if len(allFiles) > 0 else None
    
    def __init__(self, folderPath, behavior, agent, negativeReward, verbose):
        """
        The folder for the emulator should be an empty folder.
        """

        assert len(os.listdir(folderPath)) == 0, "The folder for emulator is not empty"

        self.folderPath = os.path.abspath(folderPath)
        self.behavior = behavior
        self.agent = agent
        self.verbose = verbose
        
        # set chronometer which automatically skips 10pm to 8am because it's usually when people
        # sleep
        self.chronometer = Chronometer(skipFunc=(lambda hour, _m, _d: hour < 8 or hour >= 22))
        self.stepWidthMinutes = 10
        self.numDaysPassed, self.currentHour, self.currentMinute, self.currentDay = (
                self.chronometer.forward(self.stepWidthMinutes))

        self.nextAction = MTurkEmulator.ACTION_GENERATING_BATCH_NOTIFICATION
        self.roundStartDay = 0
        self.roundEndDay = MTurkEmulator.getRoundEndDay(self.roundStartDay)

        # last notification info
        self.lastNotificationNumDays = 0
        self.lastNotificationMinute = 0
        self.lastNotificationHour = 0

        # reward information
        self.rewardCriteria = {
                ANSWER_NOTIFICATION_ACCEPT: 1,
                ANSWER_NOTIFICATION_IGNORE: 0,
                ANSWER_NOTIFICATION_DISMISS: negativeReward,
        }

        # entire results
        self.allEmulationResults = []  # a list of `EmulationRecord` objects

        # states to survey labels
        self.activityState2Labels = {
                STATE_ACTIVITY_STATIONARY: "stationary",
                STATE_ACTIVITY_WALKING: "walking",
                STATE_ACTIVITY_RUNNING: "running",
                STATE_ACTIVITY_DRIVING: "driving",
                STATE_ACTIVITY_COMMUTE: "commuting",
        }
        self.locationState2Labels = {
                STATE_LOCATION_HOME: "home",
                STATE_LOCATION_WORK: "work",
                STATE_LOCATION_OTHER: "others",
        }

    def generateNotifications(self):
        """
        Return:
          (surveyFilePath, numQuestions)
        """
        assert self.nextAction == MTurkEmulator.ACTION_GENERATING_BATCH_NOTIFICATION
        
        self.roundNotificationResults = []

        surveyFilePath = os.path.join(
                self.folderPath, "%03d-%03d.action.csv" % (self.roundStartDay, self.roundEndDay))

        with MTurkSurveyGenerator(surveyFilePath) as surveyGenerator:
            while self.numDaysPassed <= self.roundEndDay:
                # get environment info (user context)
                lastNotificationTime = utils.getDeltaMinutes(
                        self.numDaysPassed, self.currentHour, self.currentMinute,
                        self.lastNotificationNumDays, self.lastNotificationHour, self.lastNotificationMinute,
                )
                stateLastNotification = utils.getLastNotificationState(lastNotificationTime)
                stateLocation, stateActivity = self.behavior.getLocationActivity(
                        self.currentHour, self.currentMinute, self.currentDay)

                # prepare observables and get action
                stateTime = utils.getTimeState(self.currentHour, self.currentMinute)
                stateDay = utils.getDayState(self.currentDay)
                stateAll = (stateTime, stateDay, stateLocation, stateActivity, stateLastNotification)
                sendNotification = self.agent.getAction(*stateAll)
                
                # emulation recrod, but keep the reward field blank
                self.roundNotificationResults.append(EmulationRecord(
                        cxtNumDaysPassed=self.numDaysPassed,
                        cxtHour=self.currentHour,
                        cxtMinute=self.currentMinute,
                        cxtDay=self.currentDay,
                        cxtLocation=stateLocation,
                        cxtActivity=stateActivity,
                        cxtLastNotificationTime=lastNotificationTime,
                        stateTuple=stateAll,
                        decisionIsSending=sendNotification,
                        reward=None,
                ))

                # generate survey question 
                if sendNotification:
                    activityLabel = self.activityState2Labels[stateActivity]
                    locationLabel = self.locationState2Labels[stateLocation]
                    surveyGenerator.add_row(self.currentHour, self.currentMinute, self.currentDay,
                            activityLabel, locationLabel, lastNotificationTime, self.numDaysPassed)
                    self.lastNotificationNumDays = self.numDaysPassed
                    self.lastNotificationHour = self.currentHour
                    self.lastNotificationMinute = self.currentMinute

                # get the next decision time point
                self.numDaysPassed, self.currentHour, self.currentMinute, self.currentDay = (
                        self.chronometer.forward(self.stepWidthMinutes))

        # update action status
        self.nextAction = MTurkEmulator.ACTION_RECEIVING_REWARDS

        numNotificationsSent = sum([r.decisionIsSending for r in self.roundNotificationResults])

        return (surveyFilePath, numNotificationsSent)

    def probeResponseFile(self):
        assert self.nextAction == MTurkEmulator.ACTION_RECEIVING_REWARDS
        return MTurkEmulator.getFile(self.folderPath, self.roundStartDay, self.roundEndDay,
                fileType='response', extension='csv')

    def processRewards(self):
        assert self.nextAction == MTurkEmulator.ACTION_RECEIVING_REWARDS

        # value preparation
        answerNotificationCriteria = {
                'Dismiss': ANSWER_NOTIFICATION_DISMISS,
                'Accept': ANSWER_NOTIFICATION_ACCEPT,
                'Later': ANSWER_NOTIFICATION_IGNORE,
        }

        # parse response file
        responseFileName = MTurkEmulator.getFile(self.folderPath,
                self.roundStartDay, self.roundEndDay, fileType='response', extension='csv')
        responseFilePath = os.path.join(self.folderPath, responseFileName)

        flaggedWorkers = set(mturk_utils.getFlaggedWorkers())

        responseResults = {}  # (numDaysPassed, hour, minute) => ANSWER_NOTIFICATION_*
        with open(responseFilePath) as f:
            for row in csv.DictReader(f):

                # if the worker is flagged, then skip
                if row['WorkerId'] in flaggedWorkers:
                    continue

                # time is (numDaysPassed, hour, minute)
                time = tuple([int(row[k]) for k
                        in ['Input.num_days_passed', 'Input.hour', 'Input.minute']])
                result = answerNotificationCriteria[row['Answer.sentiment']]

                # if multiple records share the same time, choose the first one that does not
                # indicate the notification ignored. If all the records show the notifications
                # are ignored, then keep the response as ignored.
                if time not in responseResults:
                    responseResults[time] = ANSWER_NOTIFICATION_IGNORE
                if responseResults[time] == ANSWER_NOTIFICATION_IGNORE:
                    responseResults[time] = result

        # assign reward
        for record in self.roundNotificationResults:
            if not record.decisionIsSending:
                record.reward = 0
            else:
                time = (record.cxtNumDaysPassed, record.cxtHour, record.cxtMinute)
                if time not in responseResults:
                    raise Exception("No response found at numDaysPassed=%d, hour=%d, minute=%d"
                            % time)
                userResponse = responseResults[time]
                record.reward = self.rewardCriteria[userResponse]

        # generate reward batch history
        history = [(r.stateTuple, r.decisionIsSending, r.reward)
                for r in self.roundNotificationResults]
        self.agent.feedBatchRewards(history)

        # merge the result back
        self.allEmulationResults.extend(self.roundNotificationResults)

        # update action status
        self.nextAction = MTurkEmulator.ACTION_GENERATING_BATCH_NOTIFICATION
        self.roundStartDay = self.roundEndDay + 1
        self.roundEndDay = MTurkEmulator.getRoundEndDay(self.roundStartDay)

    def generateSavepoint(self):
        """
        Returns: savepointPath
        """
        assert self.nextAction == MTurkEmulator.ACTION_RECEIVING_REWARDS

        savepointFilePath = os.path.join(
                self.folderPath, "%03d-%03d.savepoint.p" % (self.roundStartDay, self.roundEndDay))
        dill.dump(self, open(savepointFilePath, "wb"))

        return savepointFilePath

    def getRoundStartEndDays(self):
        return (self.roundStartDay, self.roundEndDay)

    def getResultAnalyzer(self):
        return EmulationResultAnalyzer(self.allEmulationResults, self.roundStartDay)

    def _updateFolderPath(self, folderPath):
        self.folderPath = os.path.abspath(folderPath)
