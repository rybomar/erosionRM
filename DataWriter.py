import gdal
import numpy as np


class DataWriter:

    def __init__(self, filePath, X, Y, geotransform):
        self.filePath = filePath
        self.dataset = None
        self.X = X
        self.Y = Y
        self.geotransform = geotransform
        self.openFile()

    def openFile(self):
        self.dataset = gdal.GetDriverByName('GTiff').Create(self.filePath, self.X, self.Y, 1, gdal.GDT_Byte)
        self.dataset.SetGeoTransform(self.geotransform)

    def writeData(self, data):
        band = self.dataset.GetRasterBand(1)
        band.WriteArray(data)

    def closeFile(self):
        self.dataset = None
