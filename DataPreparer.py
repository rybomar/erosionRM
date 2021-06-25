import numpy as np
import joblib
from RasterReader import RasterReader
from VectorReader import VectorReader


class DataPreparer:

    def __init__(self, fileWithRasterPaths):
        with open(fileWithRasterPaths) as f:
            content = f.readlines()
        # remove whitespace characters like `\n` at the end of each line and " if occurs
        paths = [x.strip().replace('"', '') for x in content]
        self.tifsPaths = paths[1:]
        self.trainingShp = paths[0]
        testTifPaths = self.getTiffs()
        rr = RasterReader(testTifPaths[0])
        self.rasterWidth = rr.X
        self.rasterHigh = rr.Y
        self.geotransform = rr.geotransform
        self.projeciton = rr.projection
        self.numberOfClasses = 0

    def getTiffs(self):
        return self.tifsPaths

    def getTestShp(self):
        return self.trainingShp

    def getModel(self):
        model = None
        if self.trainingShp[-3:] == 'bin':
            model = joblib.load(self.trainingShp)
            self.numberOfClasses = len(np.unique(model.classes_))
        return model


    def prepareTrainingData(self):
        testTifPaths = self.getTiffs()
        trainingShp = self.getTestShp()

        vr = VectorReader(trainingShp)
        vectorData = vr.readShapefile('Id')
        classNumbers = vectorData['fields']
        objectsCount = len(classNumbers)
        featuresCount = len(testTifPaths)
        allValues = np.zeros((objectsCount, featuresCount))

        t = 0
        ids = np.arange(objectsCount)
        for tif in testTifPaths:
            rr = RasterReader(tif)
            values = rr.readValuesByPoints(vectorData['geometries'])
            allValues[:, t] = values
            t = t + 1
        trainingData = {'ids': ids, 'classNums': classNumbers, 'featuresValues': allValues}
        self.numberOfClasses = len(np.unique(classNumbers))
        return trainingData

    def getNumberOfClasses(self):
        return self.numberOfClasses

    def prepareLineForClassification(self, lineNumber):
        testTifPaths = self.getTiffs()
        featuresCount = len(testTifPaths)
        allValues = np.zeros((self.rasterWidth, featuresCount))
        t = 0
        for tif in testTifPaths:
            rr = RasterReader(tif)
            values = rr.readValuesByLine(lineNumber)
            allValues[:, t] = values
            t = t + 1
        return allValues
