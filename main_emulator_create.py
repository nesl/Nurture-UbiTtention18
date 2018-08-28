import os

from agent import *
from behavior import *
from emulator.mturk_emulator import MTurkEmulator


def main():
    rootFolder = "emulator/mturk_emulator_files/"

    # agent=Q-learning, behavior=2.txt
    #emulatorFilePath = os.path.join(rootFolder, "qlearning_2txt")
    #agent = QLearningAgent2(operatingMode=BaseAgent.MODE_BATCH)
    #behavior = ExtraSensoryBehavior('behavior/data/2.txt')
    #emulator = MTurkEmulator.createEmulator(emulatorFilePath, behavior, agent)
    
    # agent=Q-learning, behavior=4.txt
    #emulatorFilePath = os.path.join(rootFolder, "qlearning_4txt")
    #agent = QLearningAgent2(operatingMode=BaseAgent.MODE_BATCH)
    #behavior = ExtraSensoryBehavior('behavior/data/4.txt')
    #emulator = MTurkEmulator.createEmulator(emulatorFilePath, behavior, agent)

    # agent=Q-learning, behavior=5.txt
    #emulatorFilePath = os.path.join(rootFolder, "qlearning_5txt")
    #agent = QLearningAgent2(operatingMode=BaseAgent.MODE_BATCH)
    #behavior = ExtraSensoryBehavior('behavior/data/5.txt')
    #emulator = MTurkEmulator.createEmulator(emulatorFilePath, behavior, agent)
    
    # agent=Q-learning, behavior=6.txt
    #emulatorFilePath = os.path.join(rootFolder, "qlearning_6txt")
    #agent = QLearningAgent2(operatingMode=BaseAgent.MODE_BATCH)
    #behavior = ExtraSensoryBehavior('behavior/data/6.txt')
    #emulator = MTurkEmulator.createEmulator(emulatorFilePath, behavior, agent)
    
    # agent=Q-learning, behavior=6.txt, v2
    emulatorFilePath = os.path.join(rootFolder, "qlearning_7txt")
    agent = QLearningAgent2(operatingMode=BaseAgent.MODE_BATCH)
    behavior = ExtraSensoryBehavior('behavior/data/6.txt')
    emulator = MTurkEmulator.createEmulator(emulatorFilePath, behavior, agent)

    # agent=contextualBandit, behavior=2.txt
    #emulatorFilePath = os.path.join(rootFolder, "contextual_2txt")
    #agent = ContextualBanditSVMAgent(operatingMode=BaseAgent.MODE_BATCH)
    #behavior = ExtraSensoryBehavior('behavior/data/2.txt')
    #emulator = MTurkEmulator.createEmulator(emulatorFilePath, behavior, agent)
    
    # agent=contextualBandit, behavior=4.txt
    #emulatorFilePath = os.path.join(rootFolder, "contextual_4txt")
    #agent = ContextualBanditSVMAgent(operatingMode=BaseAgent.MODE_BATCH)
    #behavior = ExtraSensoryBehavior('behavior/data/4.txt')
    #emulator = MTurkEmulator.createEmulator(emulatorFilePath, behavior, agent)

    # agent=contextualBandit, behavior=5.txt
    #emulatorFilePath = os.path.join(rootFolder, "contextual_5txt")
    #agent = ContextualBanditSVMAgent(operatingMode=BaseAgent.MODE_BATCH)
    #behavior = ExtraSensoryBehavior('behavior/data/5.txt')
    #emulator = MTurkEmulator.createEmulator(emulatorFilePath, behavior, agent)

    # agent=contextualBandit, behavior=6.txt
    #emulatorFilePath = os.path.join(rootFolder, "contextual_6txt")
    #agent = ContextualBanditSVMAgent(operatingMode=BaseAgent.MODE_BATCH)
    #behavior = ExtraSensoryBehavior('behavior/data/6.txt')
    #emulator = MTurkEmulator.createEmulator(emulatorFilePath, behavior, agent)


    print("Emulator created")
   
    # generate survey questions
    roundStartDay, roundEndDay = emulator.getRoundStartEndDays()
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
