#Parámetros: normalized, minValid, bands, modelos
import xarray as xr
import numpy as np
print "Excecuting generic_classifier v1"
def isin(element, test_elements, assume_unique=False, invert=False):
    "definiendo la función isin de numpy para la versión anterior a la 1.13, en la que no existe"
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique, invert=invert).reshape(element.shape)
#Calcular el compuesto de medianas para cada uno de las entradas
nbar = xarr0
nodata=-9999
medians1={}
validValues=set()
if product=="LS7_ETM_LEDAPS":
    validValues=[66,68,130,132]
elif product == "LS8_OLI_LASRC":
    validValues=[322, 386, 834, 898, 1346, 324, 388, 836, 900, 1348]

cloud_mask=isin(nbar["pixel_qa"].values, validValues)
for band in bands:
    datos=np.where(np.logical_and(nbar.data_vars[band]!=nodata,cloud_mask),nbar.data_vars[band], np.nan)
    allNan=~np.isnan(datos)
    if normalized:
        m=np.nanmean(datos.reshape((datos.shape[0],-1)), axis=1)
        st=np.nanstd(datos.reshape((datos.shape[0],-1)), axis=1)
        datos=np.true_divide((datos-m[:,np.newaxis,np.newaxis]), st[:,np.newaxis,np.newaxis])*np.nanmean(st)+np.nanmean(m)
    medians1[band]=np.nanmedian(datos,0)
    medians1[band][np.sum(allNan,0)<minValid]=np.nan
del datos


# In[7]:

from sklearn.externals import joblib


# In[21]:

#Preprocesar: 
nmed=None
nan_mask=None
for band in bands:
    b=medians1[band].ravel()
    if nan_mask is None: 
        nan_mask=np.isnan(b)
    else:
        nan_mask=np.logical_or(nan_mask, np.isnan(medians1[band].ravel()))
    b[np.isnan(b)]=np.nanmedian(b)
    if nmed is None:
        sp=medians1[band].shape
        nmed=b
    else:
        nmed=np.vstack((nmed,b))


# In[12]:

import os
model=None
for file in os.listdir(modelos):
    print file
    if file.endswith(".pkl"):
        model=file
        break
print model
if model is None:
    raise "Debería haber un modelo en la carpeta "+modelos
    
classifier=joblib.load(os.path.join(modelos, model))
result=classifier.predict(nmed.T)
result=result.reshape(sp)


# In[ ]:




# In[24]:

coordenadas = []
dimensiones =[]
xcords = {}
for coordenada in xarr0.coords:
    if(coordenada != 'time'):
        coordenadas.append( ( coordenada, xarr0.coords[coordenada]) )
        dimensiones.append(coordenada)
        xcords[coordenada] = xarr0.coords[coordenada]
valores = {"classified": xr.DataArray(result, dims=dimensiones, coords=coordenadas)}
output = xr.Dataset(valores, attrs={'crs': xarr0.crs})
for coordenada in output.coords:
    output.coords[coordenada].attrs["units"] = xarr0.coords[coordenada].units
