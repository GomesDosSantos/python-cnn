# Python Convolutional Neural Networks

[![Miniconda](https://img.shields.io/badge/miniconda-3-green)](https://docs.conda.io/en/latest/miniconda.html)
[![Python](https://img.shields.io/badge/python-3.8-green)](https://www.python.org/)
[![Tensorflow](https://img.shields.io/badge/tensorflow-2.1-green)](https://www.tensorflow.org/)
[![JupyterLab](https://img.shields.io/badge/jupyter-1.0-green)](https://jupyter.org/)

O sistema deve reconhecer áreas de talhões (unidade mínima de cultivo de uma propriedade) em um mapa, utilizando dados multitemporais, através de inteligência artificial, a interface gráfica (Web GIS), deve permitir ao usuário selecionar um intervalo de tempo e as imagens de um catálogo disponível para a região selecionada, carregando-as em bloco para não sobrecarregar o sistema e ter opção para download.

Web GIS (Web Geographic Information System): Portal de um “Sistema de Informação Geográfica” (SIG), baseado em padrão de serviços web OGC, fornecendo uma estrutura para visualização e navegação de mapas (basemaps) e de dados geográficos vetoriais e matriciais.

- Catálogo de Imagem: O Catálogo de Imagem deve possibilitar a catalogação de coleções de dados espaço-temporal, (metadados) dos satélites Landsat 8 e Sentinel-2. Obs: O catálogo de imagem também deverá fornecer interface (web API) que permitirá consultar e recuperar as cenas de satélite catalogadas. Esta interface possibilitará que o Web GIS realize pesquisas complexas, filtrando diferentes parâmetros e especificando critérios geográficos.

- Map Tile Engine: Esse componente deve produzir “map raster tile” para uma determinada cena de satélite, obedecendo ao padrão OGC WMTS. Permitindo que
usuários do Web GIS visualizem e naveguem pelas imagens sem precisar baixá-las (real time streaming). Cada map tile é uma representação visual de parte da imagem, não dos dados em si. Esses tiles geralmente são renderizados em formato pictórico (PNG ou JPEG) que podem ser exibidos em uma aplicação web.

- Download: Após a consulta às imagens de uma determinada área de interesse, o sistema permite o download de todas as cenas (com todas as suas bandas) do período selecionado pelo usuário (Pilha de imagem).

- Máscara (Mask): Neste módulo, o sistema gerará uma máscara binária com as regiões de interesse (AOI’s) para cada cena selecionada. A construção das máscaras de saída das áreas de interesse, se dão apresentando valor igual a um (1) dentro desse poligono, enquanto as demais áreas (áreas não selecionadas) apresentam valor igual a zero (0).


<p align = "center">
  <img width = "600px" src = "./docs/assets/example.png">
</p>

- Arquivo: Após a geração da máscara para cada cena (scene), é preciso armazená-la tanto para validação visual quanto para download. Com isso, nesse módulo a máscara deve ser armazenada de alguma forma que possibilite ao usuário fazer sua validação visual (próxima etapa), assim como o download para treinamento de modelos de inteligência artificial.

> Funcionamento das redes neurais convolucionais para o processamento das imagens.

**Obs.:** Instalação do Miniconda é necessária para a execução dos comandos a seguir.

### Ambiente de Desenvolvimento
```
# Montar no ambiente Linux
$ sudo apt-get update
$ sudo apt-get install python-numpy gdal-bin libgdal-dev

# Crie um novo ambiente conda com Python3+
$ conda create --name python-cnn python=3.6.9

# Ativar o ambiente
$ conda activate python-cnn

# Com o ambiente ativado instalar o ipykernel
(python-cnn) $ conda install notebook ipykernel

# Com o ipykernel criar um kernel com o python 3.5 automático
(python-cnn) $ ipython kernel install --user --name python-cnn

# Instalar o servidor Jupyter Lab
(python-cnn) $ python -m pip install jupyter

# Instalar a biblioteca gdal e basemap para processar as imagens
(python-cnn) $ conda install -c conda-forge gdal=2.4.4 decartes

# Executar o servidor em modo de desenvolvimento
(python-cnn) $ jupyter notebook
```

> **Obs.:** Pode ser que o Notebook não  reconheça o kernel instalado pelo conda, sendo assim você pode alterar manualmente `kernel >> Change Kernel >> python-cnn`.

### Ambiente de micro serviços em docker
```
# Construir a imagem Docker
$ docker build -t jupyter-python-cnn .

# Executar a imagem
$ docker run --name jupyter-python-cnn-docker -p 8890:8888 -d jupyter-python-cnn
```

### Código fonte

> Também será necessário acesso ao servidores FTP:

- [Sentinel-1 Data](http://www.dpi.inpe.br/agricultural-database/lem/dados/cenas/Sentinel1);

- [Sentinel-2 Data](http://www.dpi.inpe.br/obt/agricultural-database/lem/dados/cenas/Sentinel2);

- [Landsat-8 Data](http://www.dpi.inpe.br/obt/agricultural-database/lem/dados/cenas/Landsat8);

> Para a execução do código fonte abaixo é necessário o download da pasta [`data/`](https://fatecspgov-my.sharepoint.com/:f:/g/personal/abner_anjos_fatec_sp_gov_br/EqcawzIRy5VBhi7LqlXGtpgB-DHpZq5IG1DspKuwD59YBA?e=4LEDNC), após o download descompacte na pasta root do projeto.


```
data/
    |_ input/
        |_ train/
            |_ false/
            |_ true/
        |_ validation/
            |_ br_uf/
            |_ false/
            |_ LEM_shapes/
            |_ true/
    |_ output/
```


```python
# !pip install tensorflow numpy matplotlib pillow wget rasterio geopandas
```


```python
import os
```


```python
import tensorflow as tf
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
```


```python
# Abstração das buscas por polígonos e georasters
from services.georasters import Georaster
from services.vector import Vector
```


```python
data = Georaster('2017-06-12 08h:35m:46s','vh', 4326)
```


```python
data.openRemoteFile()
```




    True




```python
data.projection
```




    'EPSG:4326'




```python
data.downloadRemoteFile()
```




    True




```python
data.convertFileToJPG()
```




    True




```python
data.georaster.read(1)
```




    array([[nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]], dtype=float32)




```python
data.jpg
```




<p align = "center">
  <img width = "600px" src = "./docs/assets/output_12_0.png">
</p>



```python
for coords in data.geom.get('coordinates'):
    for coord in coords:
        print(data.georaster.read(1)[int(coord[1])][int(coord[0])])
```

    -18.735785
    -19.409836
    -20.503849
    -17.980326
    -18.735785



```python
shapes = Vector(4326)
```


```python
data.geom
```


```python
data_geom = shapes.shape(data.geom.get('coordinates')[0])
data_geom
```




<p align = "left">
  <img width = "50px" src = "./docs/assets/output_16_0.svg">
</p>



```python
data.geom
```




    {'type': 'Polygon',
     'coordinates': [[[-46.422421, -11.831513],
       [-46.426524, -12.598503],
       [-45.629512, -12.601568],
       [-45.627701, -11.834385],
       [-46.422421, -11.831513]]]}




```python
shapes.lem.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NM_MUNICIP</th>
      <th>CD_GEOCMU</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LUÃS EDUARDO MAGALHÃES</td>
      <td>2919553</td>
      <td>POLYGON ((-45.71038 -12.39706, -45.71422 -12.3...</td>
    </tr>
  </tbody>
</table>
</div>




```python
shapes.covers.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>area_ha</th>
      <th>Jun_2017</th>
      <th>Jul_2017</th>
      <th>Aug_2017</th>
      <th>Sep_2017</th>
      <th>Oct_2017</th>
      <th>Nov_2017</th>
      <th>Dec_2017</th>
      <th>Jan_2018</th>
      <th>Feb_2018</th>
      <th>Mar_2018</th>
      <th>Apr_2018</th>
      <th>May_2018</th>
      <th>Jun_2018</th>
      <th>Geral</th>
      <th>variacao</th>
      <th>var</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>341.632515</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>soybean</td>
      <td>soybean</td>
      <td>soybean</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>xxxxxxxsssxxx</td>
      <td>xsx</td>
      <td>s</td>
      <td>POLYGON ((391870.392 8678209.011, 390327.395 8...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>523</td>
      <td>148.290258</td>
      <td>millet</td>
      <td>millet</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>not identified</td>
      <td>uncultivated soil</td>
      <td>not identified</td>
      <td>not identified</td>
      <td>uncultivated soil</td>
      <td>millet</td>
      <td>millet</td>
      <td>uncultivated soil</td>
      <td>llxxx-x--xllx</td>
      <td>lxlx</td>
      <td>ll</td>
      <td>POLYGON ((362953.448 8648254.537, 362492.885 8...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>196.784309</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>soybean</td>
      <td>soybean</td>
      <td>soybean</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>xxxxxxxsssxxx</td>
      <td>xsx</td>
      <td>s</td>
      <td>POLYGON ((394667.970 8677930.309, 394381.632 8...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>524</td>
      <td>28.625248</td>
      <td>sorghum</td>
      <td>sorghum</td>
      <td>sorghum</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>soybean</td>
      <td>soybean</td>
      <td>soybean</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>zzzxxxxsssxxx</td>
      <td>zxsx</td>
      <td>zs</td>
      <td>POLYGON ((378784.772 8650768.854, 378340.528 8...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>369.452478</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>soybean</td>
      <td>soybean</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>uncultivated soil</td>
      <td>xxxxxxxxssxxx</td>
      <td>xsx</td>
      <td>s</td>
      <td>POLYGON ((398795.097 8680743.662, 398796.144 8...</td>
    </tr>
  </tbody>
</table>
</div>




```python
shapes.lem.plot(color = 'black', edgecolor = 'black', figsize = (8, 8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff1c3878358>




<p align = "center">
  <img width = "600px" src = "./docs/assets/output_20_1.png">
</p>


```python
shapes.covers.plot(color = 'white', edgecolor = 'black', figsize = (8, 8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff135251be0>




<p align = "center">
  <img width = "600px" src = "./docs/assets/output_21_1.png">
</p>


```python
!ls -l data/input/train/true
```

    total 1696536
    -rw-r--r-- 1 abner abner 117589085 jun  4  2018 image10.tif
    -rw-r--r-- 1 abner abner  58825766 jun  4  2018 image11.tif
    -rw-r--r-- 1 abner abner 228341788 mai 14 11:18 image1.tif
    -rw-rw-r-- 1 abner abner 231710599 mai 27 07:59 image2.tif
    -rw-r--r-- 1 abner abner 228918854 mai 11 17:15 image3.tif
    -rw-rw-r-- 1 abner abner 231557786 mai 11 10:27 image4.tif
    -rw-rw-r-- 1 abner abner 228275958 mai 11 10:27 image5.tif
    -rw-r--r-- 1 abner abner 117743705 jun  4  2018 image6.tif
    -rw-r--r-- 1 abner abner 117743683 jun  4  2018 image7.tif
    -rw-r--r-- 1 abner abner  58903076 jun  4  2018 image8.tif
    -rw-r--r-- 1 abner abner 117589063 jun  4  2018 image9.tif



```python
for i in range(11):
    if data.convertAnyFileToJPG("data/input/train/true/image{}".format(i + 1)):
        print("Converted Image {}".format(i + 1))
```

    Converted Image 1
    Converted Image 2
    Converted Image 3
    Converted Image 4
    Converted Image 5
    Converted Image 6
    Converted Image 7
    Converted Image 8
    Converted Image 9
    Converted Image 10
    Converted Image 11



```python
!ls -l data/input/train/false
```

    total 1608788
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image10.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image11.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image12.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image13.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image14.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image1.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image2.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image3.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image4.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image5.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image6.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image7.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image8.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image9.tif



```python
for i in range(14):
    if data.convertAnyFileToJPG("data/input/train/false/image{}".format(i + 1)):
        print("Converted Image {}".format(i + 1))
```

    Converted Image 1
    Converted Image 2
    Converted Image 3
    Converted Image 4
    Converted Image 5
    Converted Image 6
    Converted Image 7
    Converted Image 8
    Converted Image 9
    Converted Image 10
    Converted Image 11
    Converted Image 12
    Converted Image 13
    Converted Image 14



```python
!ls -Rl data/input/train
```

    data/input/train:
    total 1130016
    -rw-r--r-- 1 abner abner   8298184 mai 15 14:37 clip_20170612T083546_Sigma0_VH_db.jpg
    -rw-r--r-- 1 abner abner 228341788 mai 14 11:18 clip_20170612T083546_Sigma0_VH_db.tif
    -rw-rw-r-- 1 abner abner 231710599 mai 27 07:59 clip_20170612T083546_Sigma0_VV_db.tif
    -rw-rw-r-- 1 abner abner 228275958 mai 11 10:27 clip_20170624T083547_Sigma0_VH_db.tif
    -rw-rw-r-- 1 abner abner 231557786 mai 11 10:27 clip_20170624T083547_Sigma0_VV_db.tif
    -rw-r--r-- 1 abner abner 228918854 mai 11 17:15 clip_20180315T083548_Sigma0_VH_db.tif
    drwxr-xr-x 2 abner abner      4096 mai 27 11:31 false
    drwxr-xr-x 2 abner abner      4096 mai 27 11:28 true

    data/input/train/false:
    total 856928
    -rw-r--r-- 1 abner abner 835637688 mai 27 11:30 geotifs.zip
    -rw-r--r-- 1 abner abner  41848673 mai 27 11:31 jpgs.zip

    data/input/train/true:
    total 1168760
    -rw-r--r-- 1 abner abner 1151611114 mai 27 11:27 geotifs.zip
    -rw-r--r-- 1 abner abner   45190244 mai 27 11:28 jpgs.zip



```python
!ls -l data/input/validation/true
```

    total 2020720
    -rw-rw-r-- 1 abner abner 230180639 mai 27 15:27 image1.tif
    -rw-rw-r-- 1 abner abner 233288867 mai 27 15:27 image2.tif
    -rw-rw-r-- 1 abner abner 230717127 mai 27 15:24 image3.tif
    -rw-rw-r-- 1 abner abner 233672094 mai 27 15:25 image4.tif
    -rw-rw-r-- 1 abner abner 228496143 mai 27 15:19 image5.tif
    -rw-rw-r-- 1 abner abner 226850973 mai 27 15:21 image6.tif
    -rw-rw-r-- 1 abner abner 229910342 mai 27 15:21 image7.tif
    -rw-rw-r-- 1 abner abner 229693774 mai 27 15:23 image8.tif
    -rw-rw-r-- 1 abner abner 226351595 mai 27 15:23 image9.tif



```python
for i in range(9):
    if data.convertAnyFileToJPG("data/input/validation/true/image{}".format(i + 1)):
        print("Converted Image {}".format(i + 1))
```

    Converted Image 1
    Converted Image 2
    Converted Image 3
    Converted Image 4
    Converted Image 5
    Converted Image 6
    Converted Image 7
    Converted Image 8
    Converted Image 9



```python
!ls -l data/input/validation/false
```

    total 1608772
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image10.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image11.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image12.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image13.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image14.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image1.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image2.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image3.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image4.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image5.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image6.tif
    -rw-r--r-- 1 abner abner 117743711 jun  4  2018 image7.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image8.tif
    -rw-r--r-- 1 abner abner 117589091 jun  4  2018 image9.tif



```python
for i in range(14):
    if data.convertAnyFileToJPG("data/input/validation/false/image{}".format(i + 1)):
        print("Converted Image {}".format(i + 1))
```

    Converted Image 1
    Converted Image 2
    Converted Image 3
    Converted Image 4
    Converted Image 5
    Converted Image 6
    Converted Image 7
    Converted Image 8
    Converted Image 9
    Converted Image 10
    Converted Image 11
    Converted Image 12
    Converted Image 13
    Converted Image 14



```python
PATH = "data/input"
PATH
```




    'data/input'




```python
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
```


```python
train_false_dir = os.path.join(train_dir, 'false')
train_true_dir = os.path.join(train_dir, 'true')
validation_false_dir = os.path.join(validation_dir, 'false')
validation_true_dir = os.path.join(validation_dir, 'true')
```


```python
num_false_tr = len(os.listdir(train_false_dir))
num_true_tr = len(os.listdir(train_true_dir))

num_false_val = len(os.listdir(validation_false_dir))
num_true_val = len(os.listdir(validation_true_dir))

total_train = num_false_tr + num_true_tr
total_val = num_false_val + num_true_val
```


```python
print('total training false images:', num_false_tr)
print('total training true images:', num_true_tr)

print('total validation false images:', num_false_val)
print('total validation true images:', num_true_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
```

    total training false images: 28
    total training true images: 22
    total validation false images: 42
    total validation true images: 18
    --
    Total training images: 50
    Total validation images: 60



```python
batch_size = 10
epochs = 5
IMG_HEIGHT = 150
IMG_WIDTH = 150
```


```python
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
```


```python
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)
```

    Found 50 images belonging to 2 classes.



```python
val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)
```

    Found 46 images belonging to 4 classes.



```python
sample_training_images, _ = next(train_data_gen)
```


```python
# This function will plot images in the form of
# a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```


```python
plotImages(sample_training_images[:5])
```

<p align = "center">
  <img width = "600px" src = "./docs/assets/output_42_0.png">
</p>


```python
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
```


```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_3 (Conv2D)            (None, 150, 150, 16)      448
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 75, 75, 16)        0
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 75, 75, 32)        4640
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 37, 37, 32)        0
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 37, 37, 64)        18496
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 18, 18, 64)        0
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 20736)             0
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               10617344
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 513
    =================================================================
    Total params: 10,641,441
    Trainable params: 10,641,441
    Non-trainable params: 0
    _________________________________________________________________



```python
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
```

    Epoch 1/5
    5/5 [==============================] - 988s 198s/step - loss: 1.5733 - accuracy: 0.4960 - val_loss: 3.9526 - val_accuracy: 0.0000e+00


    Exception ignored in: <bound method IteratorResourceDeleter.__del__ of <tensorflow.python.data.ops.iterator_ops.IteratorResourceDeleter object at 0x7f95b8569710>>
    Traceback (most recent call last):
      File "/home/abner/miniconda3/envs/neural-networks/lib/python3.6/site-packages/tensorflow/python/data/ops/iterator_ops.py", line 538, in __del__
        handle=self._handle, deleter=self._deleter)
      File "/home/abner/miniconda3/envs/neural-networks/lib/python3.6/site-packages/tensorflow/python/ops/gen_dataset_ops.py", line 1139, in delete_iterator
        tld.op_callbacks, handle, deleter)
    KeyboardInterrupt:


    Epoch 2/5
    4/5 [=======================>......] - ETA: 56s - loss: 0.7279 - accuracy: 0.5850