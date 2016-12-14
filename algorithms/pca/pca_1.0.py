from matplotlib.mlab import PCA
from sklearn.preprocessing import normalize
from scipy.cluster.vq import kmeans2,vq
import xarray as xr
import numpy as np
#Calcular el compuesto de medianas para cada uno de las entradas
nbar = xarr0
nodata=-9999
medians1={}
cloud_mask=np.where(np.logical_and(nbar["cf_mask"].values!=2, nbar["cf_mask"].values<4), True, False)
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
cloud_mask=np.where(np.logical_and(nbar["cf_mask"].values!=2, nbar["cf_mask"].values<4), True, False)
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
