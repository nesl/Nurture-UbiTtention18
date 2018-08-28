import math
import re
import string
from string import Template


class EmulationResultAnalyzer:

    def __init__(self, result, numDays):
        self.allResults = result
        self.weeklyResults = None
        self.dailyResults = None

        self.numDays = numDays

    def __enter__(self):
        return self

    def __exit__(self, excType, excValue, excTraceback):
        if excType is not None:
            raise Exception("An error occur in MTurkSurveyGenerator")

    def getAllEmulationResults(self):
        return self.allResults

    def getEmulationResultsGroupByWeek(self):
        if self.weeklyResults is None:
            self.weeklyResults = self._getWeeklyResults()
        return self.weeklyResults

    def getEmulationResultsGroupByDay(self):
        if self.dailyResults is None:
            self.dailyResults = self._getDailyResults()
        return self.dailyResults

    def getEmulationResultsPrintingFormat(self, formatStr):
        return self._formatEmulationResults(self.allResults, formatStr)

    def getEmulatorResultsByWeekPrintingFormat(self, formatStr):
        return [self._formatEmulationResults(partition, formatStr)
                for partition in self.getEmulationResultsGroupByWeek()]
    
    def getEmulatorResultsByDayPrintingFormat(self, formatStr):
        return [self._formatEmulationResults(partition, formatStr)
                for partition in self.getEmulationResultsGroupByDay()]

    def printEmulationResults(self, formatStr):
        print(self.getEmulationResultsPrintingFormat())

    def printEmulationResultsByWeek(self, formatStr):
        for iWeek, msg in enumerate(self.getEmulatorResultsByWeekPrintingFormat(formatStr)):
            print("Week %d: %s" % (iWeek, msg))

    def printEmulationResultsByDay(self, formatStr):
        for iDay, msg in enumerate(self.getEmulatorResultsByDayPrintingFormat(formatStr)):
            print("Week %d: %s" % (iDay, msg))
    
    def _getWeeklyResults(self):
        numWeeks = math.ceil(self.numDays / 7)
        recordPartitions = [[] for _ in range(numWeeks)]
        for r in self.allResults:
            week = int(r.cxtNumDaysPassed / 7)
            recordPartitions[week].append(r)
        return recordPartitions

    def _getDailyResults(self):
        recordPartitions = [[] for _ in range(self.roundStartDay)]
        for r in self.allResults:
            recordPartitions[r.cxtNumDaysPassed].append(r)
        return recordPartitions

    def _formatEmulationResults(self, results, formatStr):
        """
        This function summarizes the (subset of) simulation results based on the `formatStr`. For
        example, if `formatStr` is "The total reward is $totalReward", `$totalReward` will be
        replaced by the total rewards of `results`.

        The supported keywords are:
          - $totalReward
          - $numNotifications
          - $numAcceptingNotifications
          - $numIgnoringNotifications
          - $numDismissingNotifications
          - $ratioAcceptingNotifications
          - $ratioIgnoringNotifications
          - $ratioDismissingNotifications
          - $ratioAcceptsExcludeIgnores
        """

        processors = {
                '$totalReward': lambda records: sum([r.reward for r in records]),
                '$numNotifications': lambda records: self._numNotifications(records),
                '$numAcceptingNotifications': lambda records: self._numAcceptingNotifications(records),
                '$numIgnoringNotifications': lambda records: self._numIgnoringNotifications(records),
                '$numDismissingNotifications': lambda records: self._numDismissingNotifications(records),
                '$ratioAcceptingNotifications': lambda records: self._getRatio(
                        self._numAcceptingNotifications(records),
                        self._numNotifications(records),
                ),
                '$ratioIgnoringNotifications': lambda records: self._getRatio(
                        self._numIgnoringNotifications(records),
                        self._numNotifications(records),
                ),
                '$ratioDismissingNotifications': lambda records: self._getRatio(
                        self._numDismissingNotifications(records),
                        self._numNotifications(records),
                ),
                '$ratioAcceptsExcludeIgnores': lambda records: self._getRatioAcceptsExcludeIgnores(records),
        }

        allSymbols = [x for x in re.split('[^a-zA-Z$]', formatStr) if x.startswith("$")]
        substituteDict = {key[1:]: processors[key](results) for key in allSymbols}
        return Template(formatStr).substitute(substituteDict)

    def _numNotifications(self, results):
        return len([r for r in results if r.decisionIsSending])

    def _numAcceptingNotifications(self, results):
        return len([r for r in results if r.isAnAcceptedNotification()])

    def _numIgnoringNotifications(self, results):
        return len([r for r in results if r.isAnIgnoredNotification()])

    def _numDismissingNotifications(self, results):
        return len([r for r in results if r.isADismissedNotification()])

    def _getRatioAcceptsExcludeIgnores(self, results):
        numAccepts = self._numAcceptingNotifications(results)
        numDismisses = self._numDismissingNotifications(results)
        return self._getRatio(numAccepts, numAccepts + numDismisses)

    def _getRatio(self, numerator, denominator, ndigits=3):
        return round(numerator / denominator, ndigits) if denominator > 0. else 0.
