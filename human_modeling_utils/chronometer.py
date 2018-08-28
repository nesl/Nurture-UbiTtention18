class Chronometer:

    def __init__(self, startNumDaysPassed=0, startHour=0, startMinute=0, refDay=0, skipFunc=None):
        """
        Params:
          startNumDaysPassed: An int to present the number of days passed
          startHour: An int to present current hour
          startMinute: An int to present current minute
          refDay: An int which possible values are 0 (Sunday) to 6 (Saturday). It states the day
                  of the week when the chronometer starts (i.e., when `numDaysPassed` is `0`).
          skipFunc: A function which takes (dayOfTheWeek, hour, minute) as arguments and returns a
                    boolean. The function returns true when the notification presenting at the given
                    time should be skipped.
        """
        self.setTime(startNumDaysPassed, startHour, startMinute)

        assert refDay in list(range(7)), "`refDay` should be an integer between 0 to 6."
        self.referenceDay = refDay

        self.skipFunc = skipFunc

    def setTime(self, numDaysPassed, hour, minute):
        assert numDaysPassed >= 0, "`numDaysPassed` should not be a negative value"
        self.curTimestamp = self._convertTimeToTimestamp(numDaysPassed, hour, minute)

    def resetTime(self):
        self.setTime(numDaysPassed=0, hour=0, minute=0)

    def forward(self, timeDeltaMinutes):
        """
        The chronometer ticks its time every `timeDeltaMinutes`. This function may forward more
        than one step if the current time has to be skipped by `skipFunc()`. It returns the next
        valid time.

        Returns:
          (curNumDaysPassed, curHour, curMinute, curDay)
        """
        while True:
            self.curTimestamp += timeDeltaMinutes
            curNumDaysPassed, curHour, curMinute, curDay = self._extrateTimeFromTimestamp(
                    self.curTimestamp)

            # if the time is invalid, we should skip and search for next time
            if self.skipFunc is not None and self.skipFunc(curHour, curMinute, curDay):
                continue

            # return the current time
            return (curNumDaysPassed, curHour, curMinute, curDay)

    def getCurrentTime(self):
        """
        Returns:
          (curNumDaysPassed, curHour, curMinute, curDay)
        """
        return self._extrateTimeFromTimestamp(self.curTimestamp)

    def _convertTimeToTimestamp(self, numDaysPassed, hour, minute):
        return numDaysPassed * 60 * 24 + hour * 60 + minute

    def _extrateTimeFromTimestamp(self, timestamp):
        tmp, minute = divmod(timestamp, 60)
        numDaysPassed, hour = divmod(tmp, 24)
        curDay = (self.referenceDay + numDaysPassed) % 7
        return (numDaysPassed, hour, minute, curDay)
