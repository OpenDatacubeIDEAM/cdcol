import xarray as xr
import numpy as np
print("Indice de agua de diferencia normalizada modificado")
	
period_green = xarr0["green"].values
period_swir = xarr0["swir1"].values
mask_nan=np.logical_or(np.isnan(period_green), np.isnan(period_swir ))
period_mndwi = (period_green-period_swir )/(period_green+period_swir ) 
period_mndwi[mask_nan]=np.nan
#Hace un clip para evitar valores extremos.
period_mndwi[period_mndwi>1]=np.nan
period_mndwi[period_mndwi<-1]=np.nan

ncoords=[]
xdims =[]
xcords={}
for x in xarr0.coords:
    if(x!='time'):
        ncoords.append( ( x, xarr0.coords[x]) )
        xdims.append(x)
        xcords[x]=xarr0.coords[x]
variables ={"mndwi": xr.DataArray(period_mndwi, dims=xdims,coords=ncoords)}
output=xr.Dataset(variables, attrs={'crs':xarr0.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=xarr0.coords[x].units