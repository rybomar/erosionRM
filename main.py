import numpy as np
import DataPreparer
import Classifier
import DataWriter


def __main__():
    preparer = DataPreparer.DataPreparer('paths.txt')
    model = preparer.getModel()
    cl = Classifier.Classifier()
    if model == None:
        trainingData = preparer.prepareTrainingData()
        cl.loadTraining(trainingData)
        cl.trainRandomForest()
    else:
        cl = Classifier.Classifier()
        cl.loadModel(model)
    numberOfClasses = preparer.getNumberOfClasses()
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
    classWriter = DataWriter.DataWriter('erosionClassification.tif', preparer.rasterWidth, preparer.rasterHigh, preparer.geotransform)
    classWriter.writeData(resultClassImg)
    classWriter.closeFile()
    probWriter = DataWriter.DataWriter('erosionProb.tif', preparer.rasterWidth, preparer.rasterHigh, preparer.geotransform)
    maxprob = np.amax(resultProbaImg, axis=2)
    probWriter.writeData((255 * maxprob.astype(np.uint8)))
    probWriter.closeFile()


if __name__ == '__main__':
    __main__()
