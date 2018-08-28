from agent import *
from environment import *
from behavior import *
from controller import Controller

def main():
    agent = AlwaysSendNotificationAgent()
    #agent = QLearningAgent()
    #agent = QLearningAgent2()
    #agent = ContextualBanditSVMAgent()
    #agent = ContextualBanditSVMProbAgent()

    #agent = SVMAgent()
    #agent.loadModel('agent/pretrained_models/classifiers/mturk_3000_m3_r1.txt')
    
    #agent = NNAgent(negRewardWeight=5)
    #agent.loadModel('agent/pretrained_models/classifiers/mturk_3000_m3_r1.txt')

    #environment = AlwaysSayOKUser()
    #environment = StubbornUser()
    #environment = LessStubbornUser()
    #environment = SurveyUser('survey/ver1_pilot/data/02.txt')
    environment = MTurkSurveyUser(filePaths=[
            'survey/ver2_mturk/results/01_1st_Batch_3137574_batch_results.csv',
            'survey/ver2_mturk/results/02_Batch_3148398_batch_results.csv',
            'survey/ver2_mturk/results/03_Batch_3149214_batch_results.csv',
    ], filterFunc=(lambda r: ord(r['rawWorkerID'][-1]) % 3 == 2))
    
    #behavior = RandomBehavior()
    #behavior = ExtraSensoryBehavior('behavior/data/2.txt')
    #behavior = ExtraSensoryBehavior('behavior/data/4.txt')
    #behavior = ExtraSensoryBehavior('behavior/data/5.txt')
    behavior = ExtraSensoryBehavior([
        'behavior/data/2.txt',
        'behavior/data/4.txt',
        'behavior/data/5.txt',
        'behavior/data/6.txt',
    ])

    simulationWeek = 20

    controller = Controller(agent, environment, behavior,
            simulationWeek=simulationWeek, negativeReward=-5.)
    results = controller.execute()

    numNotifications, numAcceptedNotis, numDismissedNotis = _getResponseRates(results)
    answerRate = numAcceptedNotis / numNotifications
    dismissRate = numDismissedNotis / numNotifications

    expectedNumDeliveredNotifications = sum([r['probOfAnswering'] for r in results])

    print("%d decision points" % len(results))
    print("%d notifications are sent:" % numNotifications)
    print("  - %d are answered (%.2f%%)"  % (numAcceptedNotis, answerRate * 100.))
    print("  - %d are dismissed (%.2f%%)"  % (numDismissedNotis, dismissRate * 100.))
    print("Expectation of total delivered notifications is %.2f" % expectedNumDeliveredNotifications)

    # weekly performance
    print("======")
    print("Weekly performance:")
    for iWeekNum in range(simulationWeek):
        weekResults = _filterByWeek(results, iWeekNum)
        weekTotalReward = sum([r['reward'] for r in weekResults])
        numNotiTotal, numNotiAccepted, numNotiDismissed = _getResponseRates(weekResults)
        answerRate = numNotiAccepted / numNotiTotal
        dismissRate = numNotiDismissed / numNotiTotal

        print("Week %d: total reward is %.1f (%d notifications, accept=%.1f%%, dismiss=%.1f%%)" %
                (iWeekNum + 1, weekTotalReward, numNotiTotal, answerRate * 100., dismissRate * 100.))

    #agent.printQTable()

def _getResponseRates(results):
    """
    Returns: (# Total events, # accepts, # dismisses)
    """
    notificationEvents = [r for r in results if r['decision']]
    numNotifications = len(notificationEvents)
    numAcceptedNotifications = len([r for r in notificationEvents if r['reward'] > 0])
    numDismissedNotifications = len([r for r in notificationEvents if r['reward'] < 0])
    return (numNotifications, numAcceptedNotifications, numDismissedNotifications)

def _filterByWeek(results, week):
    startDay = week * 7
    endDay = startDay + 7
    return [r for r in results
            if startDay <= r['context']['numDaysPassed'] and r['context']['numDaysPassed'] < endDay]


if __name__ == "__main__":
    main()
