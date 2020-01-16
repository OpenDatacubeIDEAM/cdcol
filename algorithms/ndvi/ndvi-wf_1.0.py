import xarray as xr
import numpy as np
print("Indice de vegetacion de diferencia normalizada")

period_red = xarr0["red"].values
period_nir = xarr0["nir"].values
if 'time' in xarr0.coords:
    period_red = period_red[0]
    period_nir = period_nir[0]

mask_nan=np.logical_or(np.isnan(period_red), np.isnan(period_nir))

period_nvdi = (period_nir-period_red)/ (period_nir+period_red)

period_nvdi[mask_nan]=np.nan
#Hace un clip para evitar valores extremos. 
# Comentado por Aurelio
#period_nvdi[period_nvdi>1]=1.1
#period_nvdi[period_nvdi<-1]=-1.1
period_nvdi[period_nvdi>1]=np.nan
period_nvdi[period_nvdi<-1]=np.nan


ncoords=[]
xdims =[]
xcords={}
for x in xarr0.coords:
    if(x!='time'):
        ncoords.append( ( x, xarr0.coords[x]) )
        xdims.append(x)
        xcords[x]=xarr0.coords[x]
variables ={"ndvi": xr.DataArray(period_nvdi, dims=xdims,coords=ncoords)}
output=xr.Dataset(variables, attrs={'crs':xarr0.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=xarr0.coords[x].units

print("ndvi")
print(output)
