# -*- coding: utf-8 -*-

from matplotlib.mlab import PCA
from sklearn.preprocessing import normalize
from scipy.cluster.vq import kmeans2,vq
import xarray as xr
import numpy as np
#Calcular el compuesto de medianas para cada uno de las entradas

def isin(element, test_elements, assume_unique=False, invert=False):
    "definiendo la función isin de numpy para la versión anterior a la 1.13, en la que no existe"
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique, invert=invert).reshape(element.shape)



nbar = xarr0
nodata=-9999

validValues=set()
if product=='LS7_ETM_LEDAPS_MOSAIC':
    medians1={}
    for band in bands:
        datos = nbar.data_vars[band]
        allNan = ~np.isnan(datos)
        medians1[band] = datos
        medians1[band] = np.nanmedian(datos, 0)
        medians1[band][np.sum(allNan, 0) < minValid] = np.nan
    del datos

else:
    medians1 = {}
    if product=="LS7_ETM_LEDAPS" or product=="LS5_TM_LEDAPS":
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


nbar = xarr1
nodata=-9999
medians2={}


if product=='LS7_ETM_LEDAPS_MOSAIC':
    medians2={}
    for band in bands:
        datos =nbar.data_vars[band]
        allNan = ~np.isnan(datos)
        medians2[band] = datos
        medians2[band] = np.nanmedian(datos, 0)
        medians2[band][np.sum(allNan, 0) < minValid] = np.nan
    del datos

else:
    medians2 = {}

    if product=="LS7_ETM_LEDAPS" or product=="LS5_TM_LEDAPS":
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
        medians2[band]=np.nanmedian(datos,0)
        medians2[band][np.sum(allNan,0)<minValid]=np.nan
    del datos

#Preprocesar:
nmed=None
nan_mask=None
for band in medians1:
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
    c=medians2[band].ravel()
    nan_mask=np.logical_or(nan_mask, np.isnan(c))
    c[np.isnan(c)]=np.nanmedian(c)
    nmed=np.vstack((nmed,c))
del medians1
del medians2
#PCA
r_PCA=PCA(nmed.T)
salida= r_PCA.Y.T.reshape((r_PCA.Y.T.shape[0],)+sp)
#Kmeans - 4 clases
km_centroids, kmvalues=kmeans2(r_PCA.Y,4)
#Salida:
salida[:,nan_mask.reshape(sp)]=np.nan
kmv= kmvalues.T.reshape(sp)
kmv[nan_mask.reshape(sp)]=nodata
coordenadas = []
dimensiones =[]
xcords = {}
for coordenada in xarr0.coords:
    if(coordenada != 'time'):
        coordenadas.append( ( coordenada, xarr0.coords[coordenada]) )
        dimensiones.append(coordenada)
        xcords[coordenada] = xarr0.coords[coordenada]
valores = {"kmeans": xr.DataArray(kmv, dims=dimensiones, coords=coordenadas)}
i=1
for x in salida:
    valores["pc"+str(i)]=xr.DataArray(x, dims=dimensiones, coords=coordenadas)
    i+=1
output = xr.Dataset(valores, attrs={'crs': xarr0.crs})
for coordenada in output.coords:
    output.coords[coordenada].attrs["units"] = xarr0.coords[coordenada].units
