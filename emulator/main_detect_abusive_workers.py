from datetime import datetime
import glob
import csv
import os

from utils import mturk_utils


dirPath = os.path.dirname(os.path.realpath(__file__))


# general setup
flaggedWorkers = mturk_utils.getFlaggedWorkers()

checkPeriodMins = [10, 30, 60, 180, 144000]


# helper class
class Record:
    def __init__(self, row):
        self.time = datetime.strptime(row['SubmitTime'], '%a %b %d %H:%M:%S PDT %Y')
        if len(row['WorkerId']) < 8:
            raise Exception
        self.workerID = row['WorkerId']
        if row['Answer.sentiment'] not in ['Accept', 'Dismiss', 'Later']:
            raise Exception
        self.answer = row['Answer.sentiment']

class ResponseCounter:
    def __init__(self):
        self.accepts = 0
        self.ignores = 0
        self.dismisses = 0

    def add(self, label):
        if label == 'accept':
            self.accepts += 1
        elif label == 'Later':
            self.ignores += 1
        elif label == 'Dismiss':
            self.dismisses += 1

    def isFlagged(self):
        numTotal = self.accepts + self.ignores + self.dismisses
        if numTotal < 20:
            return False
        return any([v / numTotal > 0.8 for v in [self.accepts, self.ignores, self.dismisses]])

    def getPrintableDetail(self):
        return "%d accepts, %d laters, %d dismisses" % (self.accepts, self.ignores, self.dismisses)

# parse all the csv files
files = glob.glob(dirPath + '/mturk_emulator_files/*/[0-9][0-9][0-9]-[0-9][0-9][0-9].response*.csv')
allRecords = []

for fileName in files:
    with open(fileName) as f:
        for rawRow in csv.DictReader(f):
            try:
                record = Record(rawRow)
                allRecords.append(record)
            except:
                pass

print("Get %d rows in total" % len(allRecords))

# Show different dashboards
now = datetime.now()
for checkLengthMinutes in checkPeriodMins:
    filteredRecords = [r for r in allRecords
            if (now - r.time).total_seconds() < checkLengthMinutes * 60]
    workerDict = {}
    for record in filteredRecords:
        if record.workerID not in workerDict:
            workerDict[record.workerID] = ResponseCounter()
        workerDict[record.workerID].add(record.answer)

    print()
    print()
    print("========== Dashboard for the past %d minutes =============" % checkLengthMinutes)
    numFlagged = 0
    for worker in workerDict:
        if workerDict[worker].isFlagged():
            numFlagged += 1
            workerTxt = ("*" if worker in flaggedWorkers else " ") + worker
            print("\t%20s - %s" % (workerTxt, workerDict[worker].getPrintableDetail()))
    if numFlagged == 0:
        print("\t(none detected)")
