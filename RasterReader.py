import struct

import gdal
import numpy as np

class RasterReader(object):
    filePath = ''
    band = 0
    dataset = None
    X = 0
    Y = 0
    geotransform = None

    def __init__(self, filePath):
        self.filePath = filePath
        self.band = None
        self.geotransform = None
        self.openFile()

    def openFile(self):
        print(self.filePath)
        self.dataset = gdal.Open(self.filePath)
        self.band = self.dataset.GetRasterBand(1)
        self.geotransform = self.dataset.GetGeoTransform()
        self.X = self.dataset.RasterXSize
        self.Y = self.dataset.RasterYSize

    def getPxCoordsFromXY(self, x, y):
        xpx = int((x - self.geotransform[0]) / self.geotransform[1])
        ypx = int((y - self.geotransform[3]) / self.geotransform[5])
        return [xpx, ypx]

    def readValuesByLine(self, lineNumber):
        scanLine = self.band.ReadRaster(xoff=0, yoff=lineNumber,
                                         xsize=self.X, ysize=1,
                                         buf_xsize=self.X, buf_ysize=1,
                                         buf_type=gdal.GDT_Float32)
        tuple_of_floats = struct.unpack('f' * self.X, scanLine)
        return tuple_of_floats

    def readValueByPoint(self, xypx):
        scanpoint = self.band.ReadRaster(xoff=xypx[0], yoff=xypx[1],
                                   xsize=1, ysize=1,
                                   buf_xsize=1, buf_ysize=1,
                                   buf_type=gdal.GDT_Float32)
        tuple_of_floats = struct.unpack('f', scanpoint)
        return tuple_of_floats

    def readValuesByPoints(self, pointsGeometries):
        count = len(pointsGeometries)
        values = np.zeros(count)

        for c in range(count):
            x = pointsGeometries[c, 0]
            y = pointsGeometries[c, 1]
            xypx = self.getPxCoordsFromXY(x, y)
            value = self.readValueByPoint(xypx)
            values[c] = value[0]
        return values

