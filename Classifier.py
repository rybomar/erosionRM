import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib


class Classifier:

    def __init__(self):
        self.trainingData = None
        self.ids = None
        self.classNumbers = None
        self.featuresValues = None
        self.importanes = []
        self.bestModel = []

    def loadTraining(self, trainingData):
        self.trainingData = trainingData
        self.ids = trainingData['ids']
        self.classNumbers = trainingData['classNums']
        self.featuresValues = trainingData['featuresValues']

    def loadModel(self, model):
        self.bestModel = model
        self.classNumbers = self.bestModel.classes_

    def getNumberOfClasses(self):
        return len(np.unique(self.classNumbers))

    def trainRandomForest(self):
        maxDepth = 10
        nEstimators = 3
        foldCut = 0.25
        X_trainStart = self.featuresValues
        y_trainStart = self.classNumbers
        X_train, X_test, y_train, y_test = train_test_split(X_trainStart, y_trainStart, test_size=foldCut)
        self.bestModel = RandomForestClassifier(max_depth=maxDepth, n_estimators=nEstimators, n_jobs=4)
        self.bestModel.fit(X_train, y_train)
        y_pred, proba = self.classifyByModel(X_test)
        oa = accuracy_score(y_test, y_pred)
        print('Approx OA:' + str(oa * 100) + '%')
        self.bestModel.fit(X_trainStart, y_trainStart)
        self.importances = self.bestModel.feature_importances_
        self.saveImportances()
        self.saveModel()

    def classifyByModel(self, featuresValues):
        y_pred = self.bestModel.predict(featuresValues)
        proba = self.bestModel.predict_proba(featuresValues)
        return y_pred, proba

    def saveImportances(self):
        file = open('importances.txt', 'w')
        for imp in self.importances:
            s = str(imp)
            file.write(s)
        file.close()

    def saveModel(self):
        modelFilePath = 'maxModel.bin'
        joblib.dump(self.bestModel, modelFilePath)