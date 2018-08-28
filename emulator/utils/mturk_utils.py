import os


def getFlaggedWorkers():
    dirPath = os.path.dirname(os.path.realpath(__file__))
    filePath = os.path.join(dirPath, '../mturk_emulator_files', 'abusive_worker_list.txt')
    with open(filePath) as f:
        workerIDs = [l.strip() for l in f.readlines()]
    return workerIDs
