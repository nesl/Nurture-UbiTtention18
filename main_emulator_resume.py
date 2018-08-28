import argparse

from emulator.mturk_emulator import MTurkEmulator

def yesNoInput(desp):
    ret = input(desp + " ([y]/n): ")
    x = ret.strip()
    return len(x) == 0 or x.lower()[0] == 'y'

def main():
    parser = argparse.ArgumentParser(description="Resume an MTurk emulator")
    parser.add_argument('folder', type=str, help="The path of the emulator folder")

    args = parser.parse_args()

    emulator = MTurkEmulator.restoreEmulator(args.folder)
    #emulator.agent.printQTable()

    # check if reward file exists
    responsePath = emulator.probeResponseFile()
    if responsePath is None:
        roundStartDay, roundEndDay = emulator.getRoundStartEndDays()
        print(("The response file for day %d to day %d does not exist. Please place the response " +
               "under the emulator folder with the file name \"%03d-%03d.response.csv\"") %
                (roundStartDay, roundEndDay, roundStartDay, roundEndDay))
        exit(0)
    
    # ask user if she wants to proceed
    res = yesNoInput("The response file \"%s\" is found. Should we process it?" % responsePath)
    if not res:
        exit(0)
    
    # feed the reward file
    emulator.processRewards()

    # emulator status
    print()
    print("The following is the weekly result:")
    with emulator.getResultAnalyzer() as analyzer:
        analyzer.printEmulationResultsByWeek(
                "Total rewards: $totalReward ($numNotifications notifications sent, ratio: $ratioAcceptsExcludeIgnores)")

   	# generate survey questions
    roundStartDay, roundEndDay = emulator.getRoundStartEndDays()
    print()
    print("Generating notification questions for day %d to day %d..."
            % (roundStartDay, roundEndDay))
    surveyPath, numNotifications = emulator.generateNotifications()
    print("%d notifications have been generated." % numNotifications)
    print("The survey file is saved at %s" % surveyPath)

    # generate save point
    savepointPath = emulator.generateSavepoint()
    print()
    print("We have to take a pause here. Please upload the survey file to mTurk, " +
          "and place the mTurk result under the emulator folder.")
    print("The savepoint file is at %s" % savepointPath)

if __name__ == "__main__":
    main()
