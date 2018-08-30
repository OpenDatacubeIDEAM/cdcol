import xarray as xr
import numpy as np

def isin(element, test_elements, assume_unique=False, invert=False):
    "definiendo la función isin de numpy para la versión anterior a la 1.13, en la que no existe"
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)


nbar = xarr0
nodata=-9999
bands=["red","nir"]

validValues=set()
if product=="LS7_ETM_LEDAPS":
    validValues=[66,68,130,132]
elif product == "LS8_OLI_LASRC":
    validValues=[322, 386, 834, 898, 1346, 324, 388, 836, 900, 1348]

cloud_mask=isin(nbar["pixel_qa"].values, validValues)

datos_red=np.where(np.logical_and(nbar.data_vars["red"]!=nodata,cloud_mask),nbar.data_vars["red"], np.nan)
datos_nir=np.where(np.logical_and(nbar.data_vars["nir"]!=nodata,cloud_mask),nbar.data_vars["nir"], np.nan)

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
print max_ndvi

max_ndvi[max_ndvi>1]=np.nan
max_ndvi[max_ndvi<-1]=np.nan

import xarray as xr
ncoords=[]
xdims =[]
xcords={}
for x in nbar.coords:
    if(x!='time'):
        ncoords.append( ( x, nbar.coords[x]) )
        xdims.append(x)
        xcords[x]=nbar.coords[x]
variables ={"MAX_NDVI": xr.DataArray(max_ndvi, dims=xdims,coords=ncoords)}
            
output=xr.Dataset(variables, attrs={'crs':nbar.crs})

for x in output.coords:
    output.coords[x].attrs["units"]=nbar.coords[x].units
