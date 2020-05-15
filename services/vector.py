import geopandas
import shapely
import pyproj

# Criar uma classe para a abstração da busca por polígonos
class Vector:
    def __init__(self, projection):
        try:
            self.projection = "EPSG:{}".format(str(projection))
            self.lem = geopandas.read_file("data/input/validation/LEM_shapes/luiz_eduardo.shp")
            self.lem.crs = self.projection
            self.covers = geopandas.read_file("data/input/validation/LEM_shapes/LEM_2017_2018_mensal_training.shp")
            self.covers.crs = self.projection
        except:
            pass

    def openVector(self, name):
        try:
            return geopandas.read_file("data/input/" + name)
        except:
            return None

    def shape(self, array):
        try:
            coordinates = []
            for coords in array:
                coordinates.append((coords[0], coords[1]))
            poligono = shapely.geometry.Polygon(coordinates)
            poligono.crs = self.projection
            return poligono
        except:
            return None

