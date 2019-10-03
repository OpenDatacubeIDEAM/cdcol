#!/usr/bin/env python
# coding: utf-8

# 
# ### Adapatación código GEE

# ### Parámetros

# In[1]:


execID=1
algorithm = "RandomForestReg"
version= "1.0"


# In[2]:


products = ['LS8_OLI_LASRC'] #Productos sobre los que se hará la consulta (unidades de almacenamiento)
bands=["red","nir", "swir1","swir2"] #arreglo de bandas #"blue","green",
time_ranges = [("2016-01-01", "2016-02-25")] #Una lista de tuplas, cada tupla representa un periodo
#área sobre la cual se hará la consulta:
min_long = -73
min_lat = 11
max_long = -72
max_lat = 12


# In[3]:


normalized=True
minValid=1;

nodata=-9999


# ### Librerías

# In[4]:


import datacube
from datacube.storage import netcdf_writer
from datacube.model import Variable, CRS
import os
import re
import xarray as xr
import numpy as np
import gdal
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score


# In[6]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.utils.multiclass import unique_labels
import datetime
import matplotlib.pyplot as plt


# ### Funciones

# In[7]:


##ULTIMA FUNCION DEFINIDA
def rasterizar_entrenamiento(file_Shape,  pValFilter, pAttrFil, pAttrAGB,rows, cols, geo_transform, projection):
        labeled_pixels = np.zeros((rows, cols)) # imagen base de zeros donde empieza a llenar
        dataSource = gdal.OpenEx(file_Shape, gdal.OF_VECTOR)
        #layer = dataSource.GetLayer(0)
        layer.SetAttributeFilter(pAttrFil + " <> " + "'" + str(pValFilter) + "'")
        print(layer.GetFeatureCount())
        pClasesAGB = []
        #lee todos los poligonos para extraer el numero de clases en el arreglo 
        for feature in layer:
            pClasesAGB.append(feature.GetField(pAttrAGB))
        pClasesAGB = list(dict.fromkeys(pClasesAGB)) # remueve Duplicados
        print(pClasesAGB)
        pClase = 0 
        for val in pClasesAGB:
            layer.SetAttributeFilter(pAttrAGB + " = " + "'" + str(val)) 
            print("AGB:", val , "nroPol:", layer.GetFeatureCount())
            driver = gdal.GetDriverByName('MEM')
            target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
            target_ds.SetGeoTransform(geo_transform)
            target_ds.SetProjection(projection)
            gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[val]) ## Asigna el valor de label al poligono 
            band = target_ds.GetRasterBand(1)
            labeled_pixels += band.ReadAsArray()
        return labeled_pixels


# In[8]:


def ValidarClass(plyr, pValFilter, pAttrFil, pAttrAGB, geo_transform, projection, kFoldImg):
        rows, cols = kFoldImg.shape
        labeled_pixels = np.zeros((rows, cols)) # imagen base de zeros donde empieza a llenar
        layer.SetAttributeFilter(pAttrFil + " = " + "'" + str(pValFilter) + "'")  #### Con este filtra
        print("Nro Poligonos para el Kfold", str(pValFilter), " = ", layer.GetFeatureCount())
        pClasesAGB = []
        for feature in layer:
             pClasesAGB.append(feature.GetField(pAttrAGB))
        pClasesAGB = list(dict.fromkeys(pClasesAGB)) # remueve Duplicados si hay dos o mas poligonos con el mismo AGB
        for val in pClasesAGB:
            layer.SetAttributeFilter(pAttrAGB + " = " + str(val))  #### Con este filtra
            print("AGB:", val , "nroPol:", layer.GetFeatureCount())
            driver = gdal.GetDriverByName('MEM')
            target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
            target_ds.SetGeoTransform(geo_transform)
            target_ds.SetProjection(projection)
            gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[val]) ## Asigna el valor de label al poligono 
            band = target_ds.GetRasterBand(1)
            ### Valida el poligono contra los datos del shape
            pImgPol = np.array(band.ReadAsArray())
            pClassImaKfold = np.array(kFoldImg)
            n_samples = rows*cols
            imagRes = np.where(pImgPol.reshape((n_samples, 1)) > 0, pClassImaKfold.reshape((n_samples, 1)), 0) 
            print("AGB en Shp: ", str(val),"Media Calculada:", np.mean(imagRes[imagRes != 0]))
            labeled_pixels += imagRes.reshape((rows, cols))
        return labeled_pixels


# In[ ]:





# In[9]:


def exportar(fname, data, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None


# In[12]:


dc = datacube.Datacube(app="{}_{}_{}".format(algorithm,version,execID))


# ### Datos de entrenamiento

# In[13]:


train_data_path = '/home/cubo/jupyter/Biomasa/ipcc'


#  ## Consulta base de datos CDCol -IDEAM

# ### Máscara de nubes
# 
# Con el nuevo formato, los valores de `pixel_qa` dependen del producto. Para crear la máscara de nubes, se determinan los valores válidos para el producto actual y se usa la banda `pixel_qa` para generar un arreglo de datos booleanos: Para cada posición, si el valor de pixel_qa está en la lista de valores válidos será `True`, en caso contrario será `False`.

# In[14]:


kwargs={}
dc = datacube.Datacube(app="{}_{}_{}".format(algorithm,version,execID))
for product in products:
    i=0
    validValues=set()
    if product=="LS7_ETM_LEDAPS":
        validValues=[66,68,130,132]
    elif product == "LS8_OLI_LASRC":
        validValues=[322, 386, 834, 898, 1346, 324, 388, 836, 900, 1348]
    for tr in time_ranges:
        _data = dc.load(product=product, longitude=(min_long, max_long), latitude=(min_lat, max_lat), time=tr)
        if len(_data.data_vars)==0:
            break
        cloud_mask=np.isin(_data["pixel_qa"].values, validValues)
        for band in bands:
            _data[band].values=np.where(np.logical_and(_data.data_vars[band]!=nodata,cloud_mask),_data.data_vars[band], np.nan)
        _undesired=list(set(_data.keys())-set(bands+['latitude','longitude','time']))
        _data=_data.drop(_undesired)
            
        if "xarr"+str(i) in kwargs:
            kwargs["xarr"+str(i)]=xr.concat([kwargs["xarr"+str(i)],_data.copy(deep=True)], 'time')
        else:
            kwargs["xarr"+str(i)]=_data
    i+=1
del _data


# In[15]:


#El algoritmo recibe los productos como xarrays en variablles llamadas xarr0, xarr1, xarr2... 
xarr0=kwargs["xarr0"]
del kwargs


# ### Consulta RADAR ALOS 

# In[16]:


_data3 = dc.load(product="ALOS2_PALSAR_MOSAIC", longitude=(min_long, max_long), latitude=(min_lat, max_lat), time=("2016-01-01", "2016-12-31"))


# In[17]:


xarr0['hh']=_data3['hh']
xarr0['hv']=_data3['hv']


# ### Cálculo Índices Radar

# In[18]:


banda1=_data3['hh']
banda2=_data3['hv']


# #### Cálculo índice 1 RDFI

# In[19]:


RDFI =(banda1-banda2)/(banda1+banda2)


# In[20]:


#LPR
s1  = (banda1+banda2)
s2  =(banda1-banda2)
s3 = 2 * (banda1 **(1/2))*(banda2 **(1/2)) * np.cos(banda1-banda2)
s4 =2 * (banda1 **(1/2))*(banda2 **(1/2)) * np.sin(banda1-banda2)
LPR =(s1+s2)/(s1-s2)


# #### Cálculo índice 2 CPR

# In[21]:


SC = 0.5*s1 - 0.5 * s4
OC = 0.5*s1 + 0.5 * s4
CPR =(SC/OC)
m = (((s2**2)+(s3**2)+(s4**2))**(1/2))/s1


# ## Compuesto temporal de medianas libre de nubes

# In[22]:


medians={} 
for band in bands:
    datos=xarr0[band].values
    allNan=~np.isnan(datos) #Una mascara que indica qué datos son o no nan. 
    if normalized: #Normalizar, si es necesario.
        #Para cada momento en el tiempo obtener el promedio y la desviación estándar de los valores de reflectancia
        m=np.nanmean(datos.reshape((datos.shape[0],-1)), axis=1)
        st=np.nanstd(datos.reshape((datos.shape[0],-1)), axis=1)
        # usar ((x-x̄)/st) para llevar la distribución a media 0 y desviación estándar 1, 
        # y luego hacer un cambio de espacio para la nueva desviación y media. 
        datos=np.true_divide((datos-m[:,np.newaxis,np.newaxis]), st[:,np.newaxis,np.newaxis])*np.nanmean(st)+np.nanmean(m)
    #Calcular la mediana en la dimensión de tiempo 
    medians[band]=np.nanmedian(datos,0) 
    #Eliminar los valores que no cumplen con el número mínimo de pixeles válidos dado. 
    medians[band][np.sum(allNan,0)<minValid]=np.nan
    
del datos


# In[23]:


rows, cols = medians[bands[0]].shape


# ## Cálculo de Índices datos ópticos CDCol - IDEAM

# In[24]:


medians["ndvi"]=(medians["nir"]-medians["red"])/(medians["nir"]+medians["red"])


# In[26]:


medians["ndmi"]=(medians["nir"]-medians["swir1"])/(medians["nir"]+medians["swir1"])


# In[27]:


medians["nbr"]=(medians["nir"]-medians["swir2"])/(medians["nir"]+medians["swir2"])


# In[42]:


medians["nbr2"]=(medians["swir1"]-medians["swir2"])/(medians["swir1"]+medians["swir2"])


# In[28]:


medians["savi"]=(medians["nir"]-medians["red"])/(medians["nir"]+medians["red"]+1)*2 


# In[29]:


medians['rdfi']=RDFI[0].values


# In[30]:


medians['cpR']=CPR[0].values


# In[31]:


medians['hh']=banda1[0].values


# In[32]:


medians['hv']=banda2[0].values


# In[33]:


bands=list(medians.keys())
bands


# In[43]:


_coords=xarr0.coords
_crs=xarr0.crs


# In[44]:


rows, cols = medians[bands[0]].shape


# In[45]:


rows, cols


# In[46]:


#rows, cols = medians[bands[0]].shape
#(originX, pixelWidth, 0, originY, 0, pixelHeight)
geo_transform=(_coords["longitude"].values[0], 0.000269995,0, _coords["latitude"].values[0],0,-0.000271302)
proj=_crs.wkt
#proj='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'


# ### Clasificacion y regresion de Random Forest

# #### Preparación de insumos vector

# In[47]:


files = [f for f in os.listdir(train_data_path) if f.endswith('.shp')]
classes = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(train_data_path, f) for f in files if f.endswith('.shp')]


# In[48]:


pValFilter= 1
pAttributo = 'k10'
pAttrAGB = 'cha_HD'
labeled_pixels = np.zeros((rows, cols)) # imagen base de zeros donde empieza a llenar
dataSource = gdal.OpenEx(shapefiles[0], gdal.OF_VECTOR)
layer = dataSource.GetLayer(0)


# In[49]:


pClasesName = []
for feature in layer:
    pClasesName.append(feature.GetField(pAttributo))
    pClasesName = list(dict.fromkeys(pClasesName)) # remueve Duplicados
print(pClasesName)


# In[50]:


ImagesTrain = []
for K in pClasesName: # Recorre los ipcc o Kfold
    pima = rasterizar_entrenamiento(shapefiles[0], K, pAttributo, pAttrAGB, rows, cols, geo_transform, proj) #rasteriza todos menos el kfold
    ImagesTrain.append(pima)
    ##############con la imagen rasterizada se realiza clasificacion en esta parte del codigo para cada uno de los K-fold   ##################


# #### Iniciando clasificación Random Forest regressor

# In[51]:


start = datetime.datetime.now()  
print ('Comenzando Clasificacion: %s\n' % (start) )

kClasificaciones = []
# Parametros de la clasificacion
maxDepth=2
RandState=0
NroEstimator=10


# #### Clasificación y mapa de salida para cada kfold 

# In[52]:


for i in range(0,len(ImagesTrain)):
    print("Clasificando K-fold", i+1)
    labeled_pixels=ImagesTrain[i]
    is_train = np.nonzero(labeled_pixels)
    training_labels = labeled_pixels[is_train]
  ###
    #datosBandas = src_ds.ReadAsArray()   
  ####
    #bands_data = np.dstack(datosBandas)
    bands_data=[]
    for band in bands: 
        bands_data.append(medians[band])
    bands_data = np.dstack(bands_data)
    training_samples = bands_data[is_train]
    np.isfinite(training_samples)
    _msk=np.sum(np.isfinite(training_samples),1)>1
    training_samples= training_samples[_msk,:]
    training_labels=training_labels[_msk]
  #mascara valores nan por valor no data
    mask_nan=np.isnan(training_samples)
    training_samples[mask_nan]=-9999
  ##Clasificación RF por regresión 
    classifier = RandomForestRegressor(max_depth= maxDepth, random_state=RandState,   n_estimators=NroEstimator)
    classifier.fit(training_samples, training_labels)
    rows, cols, n_bands = bands_data.shape
    n_samples = rows*cols
    flat_pixels = bands_data.reshape((n_samples, n_bands))
  #mascara valores nan por valor no data
    mask_nan=np.isnan(flat_pixels)
    flat_pixels[mask_nan]=-9999
    flat_pixels = bands_data.reshape((n_samples, n_bands))
    result = classifier.predict(flat_pixels)
    classification = result.reshape((rows, cols))
    kClasificaciones.append(classification)


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(20,10))
pCont = 0
for ax, pIma in zip(axs, kClasificaciones):
  ax.set_title('Class Kfold ' + str(pCont))
  ax.imshow(pIma)
  pCont += 1


# In[76]:


import matplotlib.pyplot as plt
plt.imshow(ImagesVal[0])


# #### Validación de los kfold con valores de AGB del poligono

# In[53]:


#valid = ValidarClass(lyr, K, pAttributo, pAttrAGB, gt, proj, kClasificaciones[0])
ImagesVal1 = []
k=10
pContKfold = 0
for K in pClasesName: # Recorre los ipcc o Kfold
    valid = ValidarClass(shapefiles[0], K, pAttributo, pAttrAGB, geo_transform, proj, kClasificaciones[pContKfold]) #rasteriza todos menos el kfold
    ImagesVal1.append(valid)
    k=k+1
    pContKfold += 1
    print(k)


# ### Algoritmo validado y estructurado hasta aquí , las siguientes lineas se encuentran en desarrollo 

# #### Medianas de salida

# In[165]:


ImagesVal1[0].mean()


# In[ ]:


fig, axs = plt.subplots(1,3, figsize=(20,10))
pCont = 0
for ax, pIma in zip(axs, kClasificaciones):
    ax.set_title('Class Kfold ' + str(pCont))
    ax.imshow(pIma)
    pCont += 1
  


# In[ ]:


import  matplotlib.pyplot as plt  

plt.imshow(labeled_pixels)


# In[ ]:


labeled_pixels.min()


# In[ ]:


is_train = np.nonzero(labeled_pixels)
training_labels = labeled_pixels[is_train]
training_labels


# ##Listado dimensiones de insumos
# for i in  range(13):
#     medians[bands[i]].shape
#     print(medians[bands[i]].shape)

# In[ ]:


exportar("salida-class-rfr.tiff", classification, geo_transform, proj)


# ## Preparar la salida
# La salida de los algoritmos puede expresarse como: 
# - un xarray llamado output (que debe incluir entre sus atributos el crs del sistema de coordenadas)
# - un diccionario con varios xarray llamado `outputs`
# - Texto, en una variable llamada `outputtxt`

# In[ ]:


import xarray as xr
ncoords=[]
xdims =[]
xcords={}
for x in _coords:
    if(x!='time'):
        ncoords.append( ( x, _coords[x]) )
        xdims.append(x)
        xcords[x]=_coords[x]
        
variables={}
#variables ={k: xr.DataArray(v, dims=xdims,coords=ncoords)
#             for k, v in medians.items()}
variables["classification"]=xr.DataArray(kClasificaciones,dims=xdims,coords=ncoords)
output=xr.Dataset(variables, attrs={'crs':_crs})

for x in output.coords:
    _coords[x].attrs["units"]=_coords[x].units


# # Guardar la salida
# La tarea genérica se encarga de generar los archivos de salida en la carpeta adecuada. 
# 
# __Nota__: A diferencia de la tarea genérica, que maneja los 3 tipos de salida descritos en la sección anterior, este cuaderno sólo guarda la salida definida en output

# In[ ]:


from datacube.storage import netcdf_writer
from datacube.model import Variable, CRS
print "{}_{}_{}.nc".format(algorithm,version,execID)
nco=netcdf_writer.create_netcdf("{}_{}_{}.nc".format(algorithm,version,execID))
cords=('latitude', 'longitude','time')
for x in cords:
    if(x!="time"):
        netcdf_writer.create_coordinate(nco, x, _coords[x].values, _coords[x].units)
netcdf_writer.create_grid_mapping_variable(nco, _crs)
var= netcdf_writer.create_variable(nco, "classification", Variable(np.dtype(np.int32), nodata, ('latitude', 'longitude'), None) ,set_crs=True)
var[:] = netcdf_writer.netcdfy_data(classification)
nco.close()


# In[ ]:




