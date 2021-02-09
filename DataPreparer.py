import numpy as np
import joblib
from RasterReader import RasterReader
from VectorReader import VectorReader


class DataPreparer:
    rasterWidth = 0
    rasterHigh = 0
    geotransform = None
    tifsPaths = []
    trainingShp = ''
    numberOfClasses = 0

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

    def getTiffs(self):
        return self.tifsPaths
        # testTifPaths = []
        # testTifPaths.append("C:/TMP/erosion/dsm_open_pos50_ndsm.tif")
        # testTifPaths.append("C:/TMP/erosion/gradient_ndsm_c001_r03.tif")
        # testTifPaths.append("C:/TMP/erosion/std_k30_aoi.tif")
        # testTifPaths.append("C:/TMP/erosion/videbaek_1_dsm_c001_aoi_kmax_ks05.tif")
        # return testTifPaths

    def getTestShp(self):
        # trainingShp = "C:/TMP/erosion/training_2017_clip.shp"
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
