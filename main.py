import VectorReader
import RasterReader
import numpy as np
import DataPreparer
import Classifier
import DataWriter


def __main__():
    preparer = DataPreparer.DataPreparer('paths.txt')
    trainingData = preparer.prepareTrainingData()
    numberOfClasses = preparer.getNumberOfClasses()
    cl = Classifier.Classifier(trainingData)
    cl.trainRandomForest()
    resultClassImg = np.zeros((preparer.rasterHigh, preparer.rasterWidth), dtype=np.uint8)
    resultProbaImg = np.zeros((preparer.rasterHigh, preparer.rasterWidth, numberOfClasses), dtype=np.uint8)
    previousProc = -1
    for y in range(preparer.rasterHigh):
        dataLine = preparer.prepareLineForClassification(y)
        classes, proba = cl.classifyByModel(dataLine)
        resultClassImg[y, :] = classes
        resultProbaImg[y, :, :] = proba
        proc = float(y) / float(preparer.rasterHigh) * 100
        if int(proc) > previousProc:
            previousProc = int(proc)
            print(str(proc) + '%')
            # if proc >= 10:
            #     classWriter.closeFile()
            #     return
    classWriter = DataWriter.DataWriter('erosionClassification.tif', preparer.rasterWidth, preparer.rasterHigh, preparer.geotransform)
    classWriter.writeData(resultClassImg)
    classWriter.closeFile()
    probWriter = DataWriter.DataWriter('erosionProb.tif', preparer.rasterWidth, preparer.rasterHigh, preparer.geotransform)
    maxprob = np.amax(resultProbaImg, axis=2)
    probWriter.writeData((255 * maxprob.astype(np.uint8)))
    probWriter.closeFile()


if __name__ == '__main__':
    __main__()
