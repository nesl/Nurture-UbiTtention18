import sys
import csv
import numpy as np
from collections import Counter

from constant import *
from human_modeling_utils import utils
from .base_environment import BaseEnvironment


class MTurkSurveyUser(BaseEnvironment):
    """
    SurveyUser behaves based on the survey results from the mTurk. The user's mental status is
    determined based on the following strategy: Given the location, activity, weekday or weekend,
    and last notification response time, we filter out the relevant records. We assign a weight
    for each record based on the inverse of the time delta.
    """

    def __init__(self, filePaths, filterFunc=None, dismissWarningMsg=False):
        # self.behavior is a dictionary of lists. The key is the state excluding time. The values
        # are the relavent records
        self.behavior = {}
        for sDay in utils.allDayStates():
            for sLocation in utils.allLocationStates():
                for sActivity in utils.allActivityStates():
                    for sNotification in utils.allLastNotificationStates():
                        state = (sDay, sLocation, sActivity, sNotification)
                        self.behavior[state] = []

        self.records = []
        for filePath in filePaths:
            self.records.extend(self._parseFile(filePath))

        # apply the filter
        if filterFunc:
            self.records = list(filter(filterFunc, self.records))

        # arrange the records to the correct category in self.behavior
        for r in self.records:
            state = (r['stateDay'], r['stateLocation'], r['stateActivity'], r['stateNotification'])
            self.behavior[state].append(r)

        self.numNoDataStates = 0
        for state in self.behavior:
            if len(self.behavior[state]) == 0:
                sDay, sLocation, sActivity, sNotification = state
                if not dismissWarningMsg:
                    sys.stderr.write("No record for day=%d, location=%d, activity=%d, notification=%d\n"
                            % (sDay, sLocation, sActivity, sNotification))
                self.numNoDataStates += 1

        # display warning message
        if not dismissWarningMsg:
            if self.numNoDataStates > 0:
                sys.stderr.write("WARNING: No records for %d states. The behavior will be random.\n"
                        % self.numNoDataStates)

    def getResponseDistribution(self, hour, minute, day,
            stateLocation, stateActivity, lastNotificationTime):
        stateDay = utils.getDayState(day)
        stateNotification = utils.getLastNotificationState(lastNotificationTime)
        state = (stateDay, stateLocation, stateActivity, stateNotification)

        records = self.behavior[state]
        
        if len(records) == 0:
            probAnswerNotification = 0.1
            probIgnoreNotification = 0.8
            probDismissNotification = 0.1
        else:
            timeDiffs = [abs(utils.getDeltaMinutes(0, hour, minute, 0, r['rawHour'], r['rawMinute']))
                    for r in records]
            weights = np.array([(1. / (t + 5.)) ** 1 for t in timeDiffs])
            weightSum = np.sum(weights)
            probs = weights / weightSum

            chosenRecord = np.random.choice(a=records, p=probs)

            probAnswerNotification, probIgnoreNotification, probDismissNotification = 0.0, 0.0, 0.0
            if chosenRecord['answerNotification'] == ANSWER_NOTIFICATION_ACCEPT:
                probAnswerNotification = 1.0
            elif chosenRecord['answerNotification'] == ANSWER_NOTIFICATION_IGNORE:
                probIgnoreNotification = 1.0
            elif chosenRecord['answerNotification'] == ANSWER_NOTIFICATION_DISMISS:
                probDismissNotification = 1.0

        return (probAnswerNotification, probIgnoreNotification, probDismissNotification)

    def getNumTotalRecords(self):
        return len(self.records)

    def getNumNoDataStates(self):
        return self.numNoDataStates

    def getNumRecordsAcceptingNotification(self):
        return len([r for r in self.records
                if r['answerNotification'] == ANSWER_NOTIFICATION_ACCEPT])

    def getNumRecordsIgnoringNotification(self):
        return len([r for r in self.records
                if r['answerNotification'] == ANSWER_NOTIFICATION_IGNORE])

    def getNumRecordsDismissingNotification(self):
        return len([r for r in self.records
                if r['answerNotification'] == ANSWER_NOTIFICATION_DISMISS])

    def getNumUniqueWorkers(self):
        return len(set([r['rawWorkerID'] for r in self.records]))
    
    def getResponsesGroupByWorkers(self):
        """
        Return a dictionary whose keys are worker IDs, and values are tuples with the following
        format: (numTotalResponses, numAccepts, numIgnores, numDismisses).
        """
        workerAcceptingCounts = Counter([r['rawWorkerID'] for r in self.records
                if r['answerNotification'] == ANSWER_NOTIFICATION_ACCEPT])
        workerIgnoringCounts = Counter([r['rawWorkerID'] for r in self.records
                if r['answerNotification'] == ANSWER_NOTIFICATION_IGNORE])
        workerDismissingCounts = Counter([r['rawWorkerID'] for r in self.records
                if r['answerNotification'] == ANSWER_NOTIFICATION_DISMISS])
        allWorkers = set([r['rawWorkerID'] for r in self.records])

        results = {}
        for worker in allWorkers:
            numAccepts = (workerAcceptingCounts[worker]
                    if worker in workerAcceptingCounts else 0)
            numIgnores = (workerIgnoringCounts[worker]
                    if worker in workerIgnoringCounts else 0)
            numDismisses = (workerDismissingCounts[worker]
                    if worker in workerDismissingCounts else 0)
            numTotal = numAccepts + numIgnores + numDismisses
            results[worker] = (numTotal, numAccepts, numIgnores, numDismisses)
        return results

    def getAvgWorkingTime(self):
        workingDuration = [r['rawWorkingTimeSec'] for r in self.records]
        return np.mean(workingDuration), np.std(workingDuration)

    def _parseFile(self, filename):
        """
        This function receives a csv file obtained from mTurk and convert it to a list of
        dictionary objects. Please see `_parseLine()` for the format of the dictionary object.
        """
        with open(filename) as f:
            reader = csv.DictReader(f)
            records = [self._parseCsvRow(row) for row in reader]
        
        return [r for r in records if r is not None]

    def _parseCsvRow(self, row):
        """
        This function receives a line from the input file and convert it to a dictionary with
        the following keys:

            parsedRow, rawHour, rawMinute,
            rawWorkerID, rawWorkingTimeSec,
            stateDay, stateLocation, stateActivity, stateNotification,
            answerNotification

        If the line is not able to converted, or the response is invalid, `None` is returned
        instead.
        """

        workerID = row['WorkerId']
        workingTimeSec = int(row['WorkTimeInSeconds'])
        hour = int(row['Input.hour'])
        minute = int(row['Input.minute'])
        day = int(row['Input.day'])
        location = row['Input.location']
        activity = row['Input.motion']
        lastSeenNotificationTime = int(row['Input.last_notification_time'])
        response = row['Answer.sentiment']

        answerNotificationCriteria = {
            'Dismiss': ANSWER_NOTIFICATION_DISMISS,
            'Accept': ANSWER_NOTIFICATION_ACCEPT,
            'Later': ANSWER_NOTIFICATION_IGNORE,
            'Invalid': None,
        }
        answerNotification = answerNotificationCriteria[response]
        if answerNotification is None:
            return None

        locationToState = {
            'home': STATE_LOCATION_HOME,
            'work': STATE_LOCATION_WORK,
            'beach': STATE_LOCATION_OTHER,
            'friend-house': STATE_LOCATION_OTHER,
            'restaurant': STATE_LOCATION_OTHER,
            'mall': STATE_LOCATION_OTHER,
            'gym': STATE_LOCATION_OTHER,
            'park': STATE_LOCATION_OTHER,
            'movie-theater': STATE_LOCATION_OTHER,
            'market': STATE_LOCATION_OTHER,
            'others': STATE_LOCATION_OTHER,
        }

        activityToState = {
            'stationary': STATE_ACTIVITY_STATIONARY,
            'walking': STATE_ACTIVITY_WALKING,
            'running': STATE_ACTIVITY_RUNNING,
            'driving': STATE_ACTIVITY_DRIVING,
            'biking': STATE_ACTIVITY_DRIVING,
            'train': STATE_ACTIVITY_COMMUTE,
            'bus': STATE_ACTIVITY_COMMUTE,
        }

        return {
            'parsedRow': row,
            'rawHour': hour,
            'rawMinute': minute,
            'rawWorkerID': workerID,
            'rawWorkingTimeSec': workingTimeSec,
            'stateDay': utils.getDayState(day),
            'stateLocation': locationToState[location],
            'stateActivity': activityToState[activity],
            'stateNotification': utils.getLastNotificationState(lastSeenNotificationTime),
            'answerNotification': answerNotification,
        }
