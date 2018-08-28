import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

from .base_classifier_agent import BaseClassifierAgent


class NNAgent(BaseClassifierAgent):

    def trainModel(self, dataX, dataY):
        print("Begin to train the model")

        encodedY = self._encodeDataY(dataY)
        oneHopY = to_categorical(encodedY, 3)

        model = Sequential()
        model.add(Dense(32, input_dim=dataX.shape[1], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(dataX, oneHopY, epochs=20, batch_size=16, verbose=0)

        print("Finish training the model")

        return model
    
    def saveModel(self, filepath):
        sys.stderr.write("Warning: saveModel() does not support\n")
    
    def getRewardLabel(self, model, dataXVec):
        resMat = model.predict(np.array([dataXVec]))
        resIdx = np.argmax(resMat[0])
        return self.rewardLabels[resIdx]

    def _encodeDataYScalar(self, scalarY):
        if scalarY > 0:
            return 0
        elif scalarY == 0:
            return 1
        else:
            return 2

    def _encodeDataY(self, dataY):
        encodedDataY = np.array([self._encodeDataYScalar(v) for v in dataY])
        negativeReward = -10
        negVals = [v for v in dataY if v < 0]
        if len(negVals) > 0:
            negativeReward = negVals[0]
        self.rewardLabels = [1, 0, negativeReward]
        return encodedDataY

