class EmulationRecord:
    """
    All the attributes should be treated as publicly accessable
    """
    def __init__(self, cxtNumDaysPassed, cxtHour, cxtMinute, cxtDay,
            cxtLocation, cxtActivity, cxtLastNotificationTime,
            stateTuple, decisionIsSending, reward):
        self.cxtNumDaysPassed = cxtNumDaysPassed
        self.cxtHour = cxtHour
        self.cxtMinute = cxtMinute
        self.cxtDay = cxtDay,
        self.cxtLocation = cxtLocation
        self.cxtActivity = cxtActivity
        self.cxtLastNotificationTime = cxtLastNotificationTime
        self.stateTuple = stateTuple 
        self.decisionIsSending = decisionIsSending
        self.reward = reward

    def isAnAcceptedNotification(self):
        return self.decisionIsSending and self.reward > 0
    
    def isAnIgnoredNotification(self):
        return self.decisionIsSending and self.reward == 0

    def isADismissedNotification(self):
        return self.decisionIsSending and self.reward < 0
