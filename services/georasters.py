import sys
import wget
import matplotlib.image as mpimg
from osgeo import gdal
from datetime import datetime, timedelta
from PIL import Image

# Criar uma classe para a abstração da busca por imagens
class Georaster:
    def __init__(self, path):
        self.home = str(path)

    def getGeoRaster(self, date, band, file=False, download=True, convert=True):
        data_datetime = datetime.strptime(date, '%Y-%m-%d %Hh:%Mm:%Ss')
        mask = (
            "/clip_{year}{month}{day}T{hour}{minute}{second}_Sigma0_{band}_db"
        ).format(
            year=data_datetime.strftime("%Y"),
            month=data_datetime.strftime("%m"),
            day=data_datetime.strftime("%d"),
            hour=data_datetime.strftime("%H"),
            minute=data_datetime.strftime("%M"),
            second=data_datetime.strftime("%S"),
            band=str(band).upper()
        )
        if not file:
            url = (
                "http://www.dpi.inpe.br/agricultural-database/lem/dados/cenas/Sentinel1" +
                "/{year}{month}{day}_S1A" +
                mask + '.tif'
            ).format(
                year=data_datetime.strftime("%Y"),
                month=data_datetime.strftime("%m"),
                day=data_datetime.strftime("%d"),
            )
            if download:
                wget.download(
                    url,
                    'data/input/train' + mask + '.tif'
                )
            if convert:
                options_list = [
                    '-ot Byte',
                    '-of JPEG',
                    '-b 1',
                    '-scale'
                ]
                options_string = " ".join(options_list)
                try:
                    gdal.Translate(
                        'data/input/train' + mask + '.jpg',
                        'data/input/train' + mask + '.tif',
                        options=options_string
                    )
                    return {
                        "base" : url,
                        "path" : 'data/input/train' + mask + '.jpg',
                        "datetime" : data_datetime,
                        "georaster" : gdal.Open('/vsicurl/{}'.format(url)),
                        "image/jpg" : Image.open('data/input/train' + mask + '.jpg')
                    }
                except ValueError:
                    return {
                        "base" : url,
                        "path" : 'data/input/train',
                        "datetime" : data_datetime,
                        "georaster" : gdal.Open('/vsicurl/{}'.format(url)),
                        "image/jpg" : None
                    }
            else:
                return {
                    "base" : url,
                    "path" : 'data/input/train',
                    "datetime" : data_datetime,
                    "georaster" : gdal.Open('/vsicurl/{}'.format(url)),
                    "image/jpg" : None
                }
        else:
            file = (
                "data/input/train" +
                mask
            )
            if convert:
                return {
                    "base" : file,
                    "path" : 'data/input/train',
                    "datetime" : data_datetime,
                    "georaster" : gdal.Open(file + '.tif'),
                    "image/jpg" : Image.open('data/input/train' + mask + '.jpg')
                }
            else:
                return {
                    "base" : file,
                    "path" : 'data/input/train',
                    "datetime" : data_datetime,
                    "georaster" : gdal.Open(file + '.tif'),
                    "image/jpg" : None
                }