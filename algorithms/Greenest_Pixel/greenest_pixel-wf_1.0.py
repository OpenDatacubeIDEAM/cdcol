import xarray as xr
import numpy as np
nodata=-9999
datos_red = xarr0["red"].values
datos_nir = xarr0["nir"].values
ndviTotal = []
for i in range(0,len(datos_red)):
    imagen1Roja = datos_red[i]
    imagen1nir = datos_nir[i]
    allNan=~np.isnan(imagen1Roja)
    mask_nan = np.logical_or(np.isnan(imagen1Roja),np.isnan(imagen1nir))
    #Calculo de ndvi a partir de bandas red y nir
    period_nvdi = np.true_divide(np.subtract(imagen1nir,imagen1Roja),np.add(imagen1nir,imagen1Roja))
    period_nvdi[mask_nan] = np.nan
    #Almacenamiento de resultados de calculo ndvi
    ndviTotal.append(period_nvdi)
    #print ndviTotal
    #print "tiempo" + str(i)
del datos_red,datos_nir

max_ndvi = np.nanmax(ndviTotal,0)

max_ndvi[max_ndvi>1]=np.nan
max_ndvi[max_ndvi<-1]=np.nan

print(max_ndvi.dtype)
ncoords=[]
xdims =[]
xcords={}
for x in xarr0.coords:
    if(x!='time'):
        ncoords.append( ( x, xarr0.coords[x]) )
        xdims.append(x)
        xcords[x]=xarr0.coords[x]
variables ={"MAX_NDVI": xr.DataArray(max_ndvi, dims=xdims,coords=ncoords)}
output=xr.Dataset(variables, attrs={'crs':xarr0.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=xarr0.coords[x].units
