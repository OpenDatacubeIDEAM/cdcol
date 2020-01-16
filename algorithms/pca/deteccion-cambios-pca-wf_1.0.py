from matplotlib.mlab import PCA
from scipy.cluster.vq import kmeans2,vq
import xarray as xr
import numpy as np

values = list(xarrs.values())
medians1 = values[0]
medians2 = values[1]
nodata=-9999
#Preprocesar:
nmed=None
nan_mask=None
for band in medians1.data_vars.keys():
    b=medians1[band].values.ravel()
    if nan_mask is None:
        nan_mask=np.isnan(b)
    else:
        nan_mask=np.logical_or(nan_mask, np.isnan(b))
    b[np.isnan(b)]=np.nanmedian(b)
    if nmed is None:
        sp=medians1[band].shape
        nmed=b
    else:
        nmed=np.vstack((nmed,b))
    c=medians2[band].values.ravel()
    nan_mask=np.logical_or(nan_mask, np.isnan(c))
    c[np.isnan(c)]=np.nanmedian(c)
    nmed=np.vstack((nmed,c))
    print ("nmed")
    print (nmed)

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
for coordenada in values[0].coords:
    if(coordenada != 'time'):
        coordenadas.append( ( coordenada, values[0].coords[coordenada]) )
        dimensiones.append(coordenada)
        xcords[coordenada] = values[0].coords[coordenada]
#valores = {"kmeans": xr.DataArray(kmv, dims=dimensiones, coords=coordenadas)}
valores = {}
i=1
for x in salida:
    valores["pc"+str(i)]=xr.DataArray(x, dims=dimensiones, coords=coordenadas)
    i+=1

valores['kmeans'] = xr.DataArray(kmv, dims=dimensiones, coords=coordenadas)
output = xr.Dataset(valores, attrs={'crs': values[0].crs})
for coordenada in output.coords:
    output.coords[coordenada].attrs["units"] = values[0].coords[coordenada].units
