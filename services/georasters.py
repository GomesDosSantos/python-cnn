import sys
import os
import wget
import matplotlib.image as mpimg
from osgeo import gdal
from datetime import datetime, timedelta
from PIL import Image

import rasterio
import rasterio.features
import rasterio.warp

# Criar uma classe para a abstração da busca por imagens
class Georaster:
    def __init__(self, date=None, band=None, proj=None):
        self.path = "data/input/train"
        self.date_time =  datetime.strptime(date, "%Y-%m-%d %Hh:%Mm:%Ss")
        self.mask = (
            "/clip_{year}{month}{day}T{hour}{minute}{second}_Sigma0_{band}_db"
        ).format(
            year=self.date_time.strftime("%Y"),
            month=self.date_time.strftime("%m"),
            day=self.date_time.strftime("%d"),
            hour=self.date_time.strftime("%H"),
            minute=self.date_time.strftime("%M"),
            second=self.date_time.strftime("%S"),
            band=str(band).upper()
        )
        self.band = band
        self.projection = "EPSG:{}".format(str(proj))
        self.url = "http://www.dpi.inpe.br/agricultural-database/lem/dados/cenas/Sentinel1"
        self.base = (
            "http://www.dpi.inpe.br/agricultural-database/lem/dados/cenas/Sentinel1" +
            "/{year}{month}{day}_S1A" +
            self.mask + ".tif"
        ).format(
            year=self.date_time.strftime("%Y"),
            month=self.date_time.strftime("%m"),
            day=self.date_time.strftime("%d"),
        )
        self.geotiff_path = self.path
        self.jpg_path = self.path
        self.georaster = None
        self.geom = None
        self.jpg = None

    def openRemoteFile(self):
        try:
            self.georaster = rasterio.open('/vsicurl/{}'.format(self.base))
            mask = self.georaster.dataset_mask()
            for geom, val in rasterio.features.shapes(mask, transform=self.georaster.transform):
                self.geom = rasterio.warp.transform_geom(self.georaster.crs, self.projection, geom, precision=6)
            return True
        except:
            return False

    def downloadRemoteFile(self):
        file = self.path + self.mask + ".tif"
        try:
            wget.download(
                self.base,
                file
            )
            self.geotiff_path = file
            return True
        except:
            return False

    def convertFileToJPG(self):
        options_list = [
            '-ot Byte',
            '-of JPEG',
            '-b 1',
            '-scale'
        ]
        options_string = " ".join(options_list)
        root = os.path.dirname(os.path.abspath('georasters'))
        try:
            gdal.Translate(
                root + "/" + self.path + self.mask + '.jpg',
                root + "/" + self.path + self.mask + '.tif',
                options=options_string
            )
            self.jpg_path = self.path + self.mask + '.jpg'
            self.jpg = Image.open(self.jpg_path)
            return True
        except:
            return False

    def convertAnyFileToJPG(self, path):
        options_list = [
            '-ot Byte',
            '-of JPEG',
            '-b 1',
            '-scale'
        ]
        options_string = " ".join(options_list)
        root = os.path.dirname(os.path.abspath('georasters'))
        try:
            gdal.Translate(
                root + "/" + path + '.jpg',
                root + "/" + path + '.tif',
                options=options_string
            )
            return True
        except:
            return False

    def openLocalFile(self):
        try:
            self.georaster = rasterio.open(self.mask + '.tif')
            mask = self.georaster.dataset_mask()
            for geom, val in rasterio.features.shapes(mask, transform=self.georaster.transform):
                self.geom = rasterio.warp.transform_geom(self.georaster.crs, self.projection, geom, precision=6)
        except:
            return False
