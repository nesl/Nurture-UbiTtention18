import numpy as np
import itertools

from .base_behavior_model import BaseBehaviorModel
from constant import *
from collections import Counter


class ExtraSensoryBehavior(BaseBehaviorModel):
    
    def __init__(self, sourceFiles):
        """
        The `sourceFiles` is either one string indicating the path of a source file, or a `list`
        of strings that represent multiple paths.

        The file format is the following elements separated by tabs:
            <motion>
            <location>
            <second_of_the_week>
            <day_of_the_week>
            <hour_of_the_day>
            <minute_of_the_hour>

        The following is an example:
        
            stationary  home    60  0   0   1
        """

        if type(sourceFiles) is not list:
            sourceFiles = [sourceFiles]

        self.numActivityNan = 0
        self.numLocationNan = 0
        self.routings = []
        for src in sourceFiles:
            print(src)
            with open(src) as f:
                routing = [self._parseLine(l) for l in f.readlines()]
                if len(routing) == 0:
                    raise Exception("ERROR: No routing can be extracted. ('%s')" % src)
                self.routings.append(routing)

        self.srcIdx = 0
        self.recordIdx = 0

    def getLocationActivity(self, hour, minute, day):
        # handle the rewind case
        targetTime = (day, minute, hour)
        if self._getRecordTime(self.srcIdx, self.recordIdx) > targetTime:
            self.srcIdx = (self.srcIdx + 1) % len(self.routings)
            self.recordIdx = 0
            
        while self.recordIdx < len(self.routings[self.srcIdx]):
            if self._getRecordTime(self.srcIdx, self.recordIdx) >= targetTime:
                #print(self.recordIdx, self._getRecordLocationActivity(self.recordIdx))
                return self._getRecordLocationActivity(self.srcIdx, self.recordIdx)
            self.recordIdx += 1

        # the end of the routing so have to reset
        self.srcIdx = (self.srcIdx + 1) % len(self.routings)
        self.recordIdx = 0
        return self._getRecordLocationActivity(self.srcIdx, self.recordIdx)

    def _parseLine(self, line):
        """
        Please see `__init__()` for the format detail of a line.

        Returns:
          (hour, minute, day, stateActivity, stateLocation)
        """
        
        terms = line.strip().split("\t")

        if terms[0] == 'nan':
            self.numActivityNan += 1
            terms[0] = 'stationary'

        if terms[1] == 'nan':
            self.numLocationNan += 1
            terms[1] = 'others'

        activityToState = {
            'stationary': STATE_ACTIVITY_STATIONARY,
            'walking': STATE_ACTIVITY_WALKING,
            'running': STATE_ACTIVITY_RUNNING,
            'driving': STATE_ACTIVITY_DRIVING,
            'commuting': STATE_ACTIVITY_COMMUTE,
        }

        locationToState = {
            'home': STATE_LOCATION_HOME,
            'work': STATE_LOCATION_WORK,
            'others': STATE_LOCATION_OTHER,
        }

        return (
                int(terms[4]),
                int(terms[5]),
                int(terms[3]),
                locationToState[terms[1]],
                activityToState[terms[0]],
        )

    def printSummary(self):
        allRoutingRecords = list(itertools.chain(*self.routings))
        print("# of total records: %d" % len(allRoutingRecords))
        print("# of `nan` in location: %d" % self.numLocationNan)
        print("# of `nan` in Activity: %d" % self.numActivityNan)
        
        locCnts = Counter([r[3] for r in allRoutingRecords])
        print("Location:")
        print("    home: %d" % locCnts[STATE_LOCATION_HOME])
        print("    work: %d" % locCnts[STATE_LOCATION_WORK])
        print("  others: %d" % locCnts[STATE_LOCATION_OTHER])

        actCnts = Counter([r[4] for r in allRoutingRecords])
        print("Activity:")
        print("  stationary: %d" % actCnts[STATE_ACTIVITY_STATIONARY])
        print("     walking: %d" % actCnts[STATE_ACTIVITY_WALKING])
        print("     running: %d" % actCnts[STATE_ACTIVITY_RUNNING])
        print("     driving: %d" % actCnts[STATE_ACTIVITY_DRIVING])
        print("   commuting: %d" % actCnts[STATE_ACTIVITY_COMMUTE])
        
    def _getRecordTime(self, srcIdx, recordIdx):
        """
        Returns:
          (day, hour, minute)
        """
        r = self.routings[srcIdx][recordIdx]
        return (r[2], r[0], r[1])

    def _getRecordLocationActivity(self, srcIdx, recordIdx):
        """
        Returns:
          (stateLocation, stateActivity)
        """
        r = self.routings[srcIdx][recordIdx]
        return (r[3], r[4])
