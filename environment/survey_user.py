import numpy

from constant import *
from human_modeling_utils import utils
from .base_environment import BaseEnvironment


class SurveyUser(BaseEnvironment):
    """
    SurveyUser behaves based on the survey results (derived from notification_survey.html). The
    user's mental status is determined based on the following strategy: Given the location,
    activity, weekday or weekend, and last notification response time, we filter out the relevant
    records. We assign a weight for each record based on the inverse of the time delta.
    """

    def __init__(self, filePath):
        with open(filePath) as f:
            lines = f.readlines()

        records = [self.parse(l) for l in lines]
        records = [r for r in records if r is not None]

        # self.behavior is a dictionary of lists. The key is the state excluding time. The values
        # are the relavent records
        self.behavior = {}
        for sDay in utils.allDayStates():
            for sLocation in utils.allLocationStates():
                for sActivity in utils.allActivityStates():
                    for sNotification in utils.allLastNotificationStates():
                        state = (sDay, sLocation, sActivity, sNotification)
                        self.behavior[state] = []

        # arrange the records to the correct category in self.behavior
        for r in records:
            state = (r['stateDay'], r['stateLocation'], r['stateActivity'], r['stateNotification'])
            self.behavior[state].append(r)

    def getResponseDistribution(self, hour, minute, day,
            stateLocation, stateActivity, lastNotificationTime):

        stateDay = utils.getDayState(day)
        stateNotification = utils.getLastNotificationState(lastNotificationTime)
        state = (stateDay, stateLocation, stateActivity, stateNotification)

        records = self.behavior[state]
        
        if len(records) == 0:
            probAnswerNotification = 0.1
        else:
            timeDiffs = [abs(utils.getDeltaMinutes(0, hour, minute, 0, r['rawHour'], r['rawMinute']))
                    for r in records]
            weights = numpy.array([1. / (t + 5.) for t in timeDiffs])
            weightSum = numpy.sum(weights)
            probs = weights / weightSum

            chosenRecord = numpy.random.choice(a=records, p=probs)
            probAnswerNotification = (1.0 if chosenRecord['answerNotification'] else 0.0)

        probDismissNotification = 1.0 - probAnswerNotification
        probIgnoreNotification = 0.0
        return (probAnswerNotification, probIgnoreNotification, probDismissNotification)

    def parse(self, line):
        """
        This function receives a line from the input file and convert it to a dictionary with
        the following keys:

            rawLine, rawHour, rawMinute,
            stateDay, stateLocation, stateActivity, stateNotification, answerNotification

        If the line is not able to converted, or the response is invalid, None is returned instead

        Example of a line: 16,0,0,work,walking,90,answer
            - hour (0-23)
            - minute (0-59)
            - day (0-6)
            - location {home, work, others}
            - activity {stationary, walking, running, driving}
            - last seen notification {0-integer}
            - response {ignore, dismiss, answer, later, invalid}
        """

        terms = line.strip().split(',')
        if len(terms) < 7:
            return None

        hour = int(terms[0])
        minute = int(terms[1])
        day = int(terms[2])
        location = terms[3]
        activity = terms[4]
        lastSeenNotificationTime = int(terms[5])
        response = terms[6]

        answerNotificationCriteria = {
            'ignore': False,
            'dismiss': False,
            'answer': True,
            'later': None,
            'invalid': None,
        }
        answerNotification = answerNotificationCriteria[response]
        if answerNotification is None:
            return None

        locationToState = {
            'home': STATE_LOCATION_HOME,
            'work': STATE_LOCATION_WORK,
            'others': STATE_LOCATION_OTHER,
        }

        activityToState = {
            'stationary': STATE_ACTIVITY_STATIONARY,
            'walking': STATE_ACTIVITY_WALKING,
            'running': STATE_ACTIVITY_RUNNING,
            'driving': STATE_ACTIVITY_DRIVING,
        }

        return {
            'rawLine': line,
            'rawHour': hour,
            'rawMinute': minute,
            'stateDay': utils.getDayState(day),
            'stateLocation': locationToState[location],
            'stateActivity': activityToState[activity],
            'stateNotification': utils.getLastNotificationState(lastSeenNotificationTime),
            'answerNotification': answerNotification,
        }
