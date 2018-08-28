class BaseEnvironment:
    
    def getResponseDistribution(self, hour, minute, day,
            stateLocation, stateActivity, lastNotificationTime):
        """
        This function returns user's internal perception of notifications (i.e., at this given
        context, how likely the user is going to answer the notification.)

        Returns:
          (probAnsweringNotification, probIgnoringNotification, probDismissingNotification)
            - probAnsweringNotification: A float number between 0. to 1.
            - probIgnoringNotification: A float number between 0. to 1.
            - probDismissingNotification: A float number between 0. to 1.

          The sum of `probAnsweringNotification`, `probIgnoringNotification`, and
          `probDismissingNotification` is expected to be 1.0.
        """
        pass
