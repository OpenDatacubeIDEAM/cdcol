import xarray as xr
import numpy as np
print "Excecuting medianas v2"

#La funcion isin permite verificar si el elemento (elemento) que se pasa por parametro está dentro del arreglo (test_elements)
#Esta funcion se define para verificar los valores validos de la mascara de nube (pixel_qa) 
def isin(element, test_elements, assume_unique=False, invert=False):
    "definiendo la función isin de numpy para la versión anterior a la 1.13, en la que no existe"
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)
nbar = xarr0
#Un valor negativo suficientemente grande para representar cuando no hay datos
nodata=-9999
medians={}

#Conjunto de valores valido para la mascara de nube
validValues=set()

#Dependiendo del satellite, se definen unos valores distintos para la mascara
#Estos valores se definen en la página de LandSat: https://landsat.usgs.gov/landsat-surface-reflectance-quality-assessment
#Son valores validos porque representan agua (water) o un "pixel claro" (clear)
if product=="LS7_ETM_LEDAPS" or product=="LS5_TM_LEDAPS":
    validValues=[66,68,130,132]
elif product == "LS8_OLI_LASRC":
    validValues=[322, 386, 834, 898, 1346, 324, 388, 836, 900, 1348]

cloud_mask=isin(nbar["pixel_qa"].values, validValues)

#Se hace un recorrido sobre las bandas
for band in bands:
    # np.where es la funcion where de la libreria numpy que retorna un arreglo o un set de arreglos con datos que cumplen con la condicion dada por parametro
    # 
    datos=xarr0[band].values
    allNan=~np.isnan(datos) #Una mascara que indica qué datos son o no nan. 
    if normalized: #Normalizar, si es necesario.
        #Para cada momento en el tiempo obtener el promedio y la desviación estándar de los valores de reflectancia
        m=np.nanmean(datos.reshape((datos.shape[0],-1)), axis=1)
        st=np.nanstd(datos.reshape((datos.shape[0],-1)), axis=1)
        # usar ((x-x̄)/st) para llevar la distribución a media 0 y desviación estándar 1, 
        # y luego hacer un cambio de espacio para la nueva desviación y media. 
        datos=np.true_divide((datos-m[:,np.newaxis,np.newaxis]), st[:,np.newaxis,np.newaxis]) #*np.nanmean(st)+np.nanmean(m)
    #Calcular la mediana en la dimensión de tiempo 
    medians[band]=np.nanmedian(datos,0) 
#Elimina la variable datos y la asociacion que tiene en el algoritmo
del datos

import xarray as xr
ncoords=[]
xdims =[]
xcords={}
for x in nbar.coords:
    if(x!='time'):
        ncoords.append( ( x, nbar.coords[x]) )
        xdims.append(x)
        xcords[x]=nbar.coords[x]
variables ={k: xr.DataArray(v, dims=xdims,coords=ncoords)
             for k, v in medians.items()}
#Genera el dataset (netcdf) con las bandas con el sistema de referencia de coordenadas
output=xr.Dataset(variables, attrs={'crs':nbar.crs})

for x in output.coords:
    output.coords[x].attrs["units"]=nbar.coords[x].units
