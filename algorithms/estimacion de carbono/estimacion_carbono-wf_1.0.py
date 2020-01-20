import os,posixpath
import re
import xarray as xr
import numpy as np
import gdal
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm 
from sklearn.svm import SVC
import datacube
from datacube.storage import netcdf_writer
from datacube.model import Variable, CRS
import os
import re
import xarray as xr
import numpy as np
import gdal
from sklearn.datasets import make_regression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.utils.multiclass import unique_labels
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal,ogr
from glob import glob
import matplotlib as mpl


def enmascarar_entrenamiento(vector_data_path, cols, rows, geo_transform, projection, target_value=1):
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds

def rasterizar_entrenamiento(plyr, pValFilter, pAttrFil, pAttrAGB, rows, cols , geo_transform, projection):
    labeled_pixels = np.zeros((rows, cols)) # imagen base de zeros donde empieza a llenar
    plyr.SetAttributeFilter(pAttrFil + " <> " + "'" + str(pValFilter) + "'")  #### Con este filtra
    print("Nro de Pol Filtrados  <> de",str(pValFilter),":", plyr.GetFeatureCount())
    pClasesAGB = []
    for feature in plyr:
        pClasesAGB.append(feature.GetField(pAttrAGB))
        pClasesAGB = list(dict.fromkeys(pClasesAGB)) # remueve Duplicados si hay dos o mas poligonos con el mismo AGB
    print(pClasesAGB)
    for val in pClasesAGB:
        plyr.SetAttributeFilter(pAttrAGB + " = " + str(val))  #### Con este filtra
        print("AGB:", val , "nroPol:", plyr.GetFeatureCount())
        driver = gdal.GetDriverByName('MEM')
        target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
        target_ds.SetGeoTransform(geo_transform)
        target_ds.SetProjection(projection)
        gdal.RasterizeLayer(target_ds, [1], plyr, burn_values=[val]) ## Asigna el valor de label al poligono 
        band = target_ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
    return labeled_pixels

def ValidarClass(plyr, pValFilter, pAttrFil, pAttrAGB, geo_transform, projection, kFoldImg):
        rows, cols = kFoldImg.shape
        labeled_pixels = np.zeros((rows, cols)) # imagen base de zeros donde empieza a llenar
        plyr.SetAttributeFilter(pAttrFil + " = " + "'" + str(pValFilter) + "'")  #### Con este filtra
        print("Nro Poligonos para el Kfold", str(pValFilter), " = ", plyr.GetFeatureCount())
        pClasesAGB = []
        for feature in plyr:
             pClasesAGB.append(feature.GetField(pAttrAGB))
        pClasesAGB = list(dict.fromkeys(pClasesAGB)) # remueve Duplicados si hay dos o mas poligonos con el mismo AGB
        for val in pClasesAGB:
            plyr.SetAttributeFilter(pAttrAGB + " = " + str(val))  #### Con este filtra
            print("AGB:", val , "nroPol:", plyr.GetFeatureCount())
            driver = gdal.GetDriverByName('MEM')
            target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
            target_ds.SetGeoTransform(geo_transform)
            target_ds.SetProjection(projection)
            gdal.RasterizeLayer(target_ds, [1], plyr, burn_values=[val]) ## Asigna el valor de label al poligono 
            band = target_ds.GetRasterBand(1)
            ### Valida el poligono contra los datos del shape
            pImgPol = np.array(band.ReadAsArray())
            pClassImaKfold = np.array(kFoldImg)
            n_samples = rows*cols
            imagRes = np.where(pImgPol.reshape((n_samples, 1)) > 0, pClassImaKfold.reshape((n_samples, 1)), 0) 
            print("AGB en Shp: ", str(val),"Media Calculada:", np.mean(imagRes[imagRes != 0]))
            labeled_pixels += imagRes.reshape((rows, cols))
        return labeled_pixels




# The trainning data must be in a zip folder.
train_zip_file_name  = [file_name for file_name in os.listdir(train_data_path) if file_name.endswith('.zip')][0]
train_zip_file_path = os.path.join(train_data_path,train_zip_file_name)
train_folder_path = train_zip_file_path.replace('.zip','')

print('train_zip_file_path',train_zip_file_path)
print('train_folder_path',train_folder_path)

zip_file = zipfile.ZipFile(train_zip_file_path)
zip_file.extractall(train_data_path)
zip_file.close()

files = [f for f in os.listdir(train_folder_path) if f.endswith('.shp')]
classes = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(train_folder_path, f) for f in files if f.endswith('.shp')]
rows, cols = xarr0[product['bands'][0]].shape

##test1

values = list(xarr0.values())
#print('values xarr')
print('coordenadas')
print(xarr0.coords)

print('rows',rows)
print('cols',cols)

_coords=xarr0.coords

print('bandas xarr0',list(xarr0.data_vars))
lista=list(xarr0.data_vars)

geo_transform=(_coords["longitude"].values[0], 0.000269995,0, _coords["latitude"].values[0],0,-0.000271302)
proj = xarr0.crs.crs_wkt
###


data_source = gdal.OpenEx(train_folder_path, gdal.OF_VECTOR)
layer = data_source.GetLayer(0)
print('layer')
print(type(layer))
print(layer)

lyr=data_source.GetLayer()
print('lyr')
print(type(lyr))
print(lyr)

pAttributo = 'K10'
pAttrAGB = 'cha_HD'

print('test 1')


pClasesName = []
#lee todos los poligonos para extraer el numero de clases en el arreglo 

for feature in lyr:
    pClasesName.append(feature.GetField(pAttributo))
    pClasesName = list(dict.fromkeys(pClasesName)) # remueve Duplicados
print(pClasesName)

##Rasterizacion de poligonos
###Se procede a realizar  la rasterizacion de los poligonos ingresados definidos con los parametros anteriormente
ImagesTrain = []
for K in pClasesName: # Recorre los ipcc o Kfold
    pima = rasterizar_entrenamiento(lyr, K, pAttributo, pAttrAGB, rows, cols, geo_transform, proj) #rasteriza todos menos el kfold
    print(pima.max())
    ImagesTrain.append(pima)
        ##############   con la imagen rasterizada se realiza clasificacion en esta parte del codigo para cada uno de los K-fold   ##################
   
start = datetime.datetime.now()  
print ('Comenzando Clasificacion: %s\n' % (start) )

kClasificaciones = []
# Parametros de la clasificacion definidos para  la funcion Random Forest Regressor
maxDepth=2
RandState=0
NroEstimator=10

from numpy import inf


for i in range(0,len(ImagesTrain)):
    print("Clasificando K-fold", i+1)
    labeled_pixels=ImagesTrain[i]
    is_train = np.nonzero(labeled_pixels)
    training_labels = labeled_pixels[is_train]
    print("bands_data")
    bands_data=[]
    for band in lista:
    	if band != 'pixel_qa':
        	bands_data.append(xarr0[band])
    bands_data = np.dstack(bands_data)
    print(bands_data)

    training_samples = bands_data[is_train]
    np.isfinite(training_samples)
    _msk=np.sum(np.isfinite(training_samples),1)>1
    training_samples= training_samples[_msk,:]
    training_labels=training_labels[_msk]
    #asignacion de mascara valores nan por valor no data
    mask_nan=np.isnan(training_samples)
    training_samples[mask_nan]=-9999
    print('valores training_samples')
    print(training_samples.min())
    print(training_samples.max())
    print(training_samples.dtype)
    print('valores training_labels')
    print(training_labels.min())
    print(training_labels.max())
    print(training_labels.dtype)
    training_labels=training_labels.astype('float32')
    print(training_labels.dtype)
    ##Clasificacion RF por regresion con los parametros definidos anteriormente
    classifier = RandomForestRegressor(max_depth= maxDepth, random_state=RandState,   n_estimators=NroEstimator)
    classifier.fit(training_samples, training_labels)
    print('classifier')
    rows, cols, n_bands = bands_data.shape
    n_samples = rows*cols
    flat_pixels = bands_data.reshape((n_samples, n_bands))
    #asignacion de mascara valores nan por valor no data=-9999
    mask_nan=np.isnan(flat_pixels)
    flat_pixels[mask_nan]=-9999
    flat_pixels = bands_data.reshape((n_samples, n_bands))
    print(flat_pixels.min())
    print(flat_pixels.max())
    print(flat_pixels.dtype)
    flat_pixels[flat_pixels==-inf]= 0
    flat_pixels[flat_pixels>= 1E308]=0
    ##preparacion de resultados en matriz con dimensiones igual a las consultadas al principio del codigo
    result = classifier.predict(flat_pixels)
    classification = result.reshape((rows, cols))
    kClasificaciones.append(classification)



##Preparacion de salida final 
kClasificaciones=np.asarray(kClasificaciones)
ImgResultado = np.zeros((rows, cols))
#Recorrer la matriz pixel a pixel
for l in range(0,cols):
    for j in range(0,rows): 
        dato = kClasificaciones[:,j,l]
        ## Calculo de percentil 5 y 95 de los datos resultantes de cada set de datos correspondientes con las 7 clasificaciones anteriore resultantes, calculado pixel a pixel
        percent=np.percentile(dato,q=[5,95])
        #Calculo de mediana para set de datos extrayendo los valores que se encuentre <5 y > 95 de los valores correspondientes a percentil, generando un solo pixel resultante.
        datoFinal = np.median(dato[np.logical_and(dato>percent[0],dato<percent[1])])
        ImgResultado[j,l] = datoFinal

#Valor asignado para estimacion de carbono del codigo original GEE
BCR = 0.47;
result = ImgResultado*BCR


coordenadas = []
dimensiones =[]
xcords = {}
for coordenada in xarr0.coords:
    if(coordenada != 'time'):
        coordenadas.append( ( coordenada, xarr0.coords[coordenada]) )
        dimensiones.append(coordenada)
        xcords[coordenada] = xarr0.coords[coordenada]


print('asignacion result 2')

valores = {"carbono": xr.DataArray(result, dims=dimensiones, coords=coordenadas)}
#Genera el dataset (netcdf) con las bandas con el sistema de referencia de coordenadas
output = xr.Dataset(valores, attrs={'crs': xarr0.crs})

for coordenada in output.coords:
    output.coords[coordenada].attrs["units"] = xarr0.coords[coordenada].units


