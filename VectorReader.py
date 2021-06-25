import ogr
import numpy as np


class VectorReader(object):

    def __init__(self, filePath):
        self.filePath = filePath

    def readShapefile(self, columnName):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(self.filePath, 0)  # 0 means read-only. 1 means writeable.

        if dataSource is None:
            print('Could not open ' + self.filePath)
            return None
        else:
            layer = dataSource.GetLayer()
            featureCount = layer.GetFeatureCount()
            print("Number of features: " + str(featureCount))
            geometries = np.empty([featureCount, 2])
            fields = []
            counter = 0
            for feature in layer:
                geom = feature.GetGeometryRef()
                field = feature.GetField(columnName)
                geometries[counter, 0] = geom.GetX()
                geometries[counter, 1] = geom.GetY()
                fields.append(field)
                counter = counter + 1
            vectorData = {'geometries': geometries, 'fields': fields}
        return vectorData
