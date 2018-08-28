import os
import csv
import numpy as np

from utils import utils
from environment import *
from constant import *


# configuration
inPath = "emulator/mturk_emulator_files/contextual_2txt/035-041.action.csv"
environment = MTurkSurveyUser(filePaths=[
        'survey/ver2_mturk/results/01_1st_Batch_3137574_batch_results.csv',
        'survey/ver2_mturk/results/02_Batch_3148398_batch_results.csv',
        'survey/ver2_mturk/results/03_Batch_3149214_batch_results.csv',
])

# constant dictionaries
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
        'commuting': STATE_ACTIVITY_COMMUTE,
}

# check output path
folder, inFile = os.path.split(inPath)
assert len(inFile) >= 14  # ddd-ddd.action
outFile = inFile[:7] + ".response.csv"
outPath = os.path.join(folder, outFile)

# here we go
with open(outPath, 'w', newline='') as fo:
    outFieldNames = [
            'Input.content',
            'Input.hour',
            'Input.minute',
            'Input.day',
            'Input.motion',
            'Input.location',
            'Input.last_notification_time',
            'Input.num_days_passed',
            'Answer.sentiment',
    ]
    writer = csv.DictWriter(fo, fieldnames=outFieldNames)
    writer.writeheader()

    with open(inPath) as f:
        for row in csv.DictReader(f):
            # get user response
            hour = int(row['hour'])
            minute = int(row['minute'])
            day = int(row['day'])
            stateLocation = locationToState[row['location']]
            stateActivity = activityToState[row['motion']]
            lastNotificationTime = int(row['last_notification_time'])

            probAnsweringNotification, probIgnoringNotification, probDismissingNotification = (
                    environment.getResponseDistribution(
                            hour, minute, day, stateLocation, stateActivity, lastNotificationTime,
                    )
            )
            probAnsweringNotification, probIgnoringNotification, probDismissingNotification = utils.normalize(
                    probAnsweringNotification, probIgnoringNotification, probDismissingNotification)

            userReaction = np.random.choice(
                    a=['Accept', 'Later', 'Dismiss'],
                    p=[probAnsweringNotification, probIgnoringNotification, probDismissingNotification],
            )

            writer.writerow({
                    'Input.content': row['content'],
                    'Input.hour': row['hour'],
                    'Input.minute': row['minute'],
                    'Input.day': row['day'],
                    'Input.motion': row['motion'],
                    'Input.location': row['location'],
                    'Input.last_notification_time': row['last_notification_time'],
                    'Input.num_days_passed': row['num_days_passed'],
                    'Answer.sentiment': userReaction,
            }) 
