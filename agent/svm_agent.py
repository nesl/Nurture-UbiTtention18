import sys
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing

from .base_classifier_agent import BaseClassifierAgent


kTunedParameters = [{
    'kernel': ['rbf'],
    'gamma': [2**e for e in [-6, -3, 0, 3, 6]],
    'C': [2**e for e in [-6, -3, 0, 3, 6]],
}]

kFold = 5

class SVMAgent(BaseClassifierAgent):

    def trainModel(self, dataX, dataY):
        print("Begin to train the model")
        dataX.astype(float)
        scaler = preprocessing.StandardScaler().fit(dataX)
        scaledDataX = scaler.transform(dataX.astype(float))
        clf = GridSearchCV(SVC(), kTunedParameters, cv=kFold, n_jobs=8)
        clf.fit(scaledDataX, dataY)
        print("Finish training the model")
        return {
                'classifier': clf,
                'scaler': scaler,
        }
    
    def saveModel(self, filepath):
        sys.stderr.write("Warning: saveModel() does not support\n")
    
    def getRewardLabel(self, model, dataXVec):
        clf = model['classifier']
        scaler = model['scaler']
        
        scaledDataX = scaler.transform([dataXVec])
        res = clf.predict(scaledDataX)
        return res[0] > 0
