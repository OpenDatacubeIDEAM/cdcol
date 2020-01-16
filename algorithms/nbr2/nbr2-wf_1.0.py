# Normalized Burn Ratio(NBR2)
import xarray as xr
import numpy as np
print("Indice de estimación de severidad de fuegos Version 2")


period_swir1 = xarr0["swir1"].values
period_swir2 = xarr0["swir2"].values

print("mascara valores nan")
mask_nan=np.logical_or(np.isnan(period_swir1), np.isnan(period_swir2))
period_nbr2 = (period_swir1-period_swir2)/(period_swir1+period_swir2) 
period_nbr2[mask_nan]=np.nan
#Hace un clip para evitar valores extremos
period_nbr2[period_nbr2>1]=np.nan
period_nbr2[period_nbr2<-1]=np.nan

ncoords=[]
xdims =[]
xcords={}
for x in xarr0.coords:
    if(x!='time'):
        ncoords.append( ( x, xarr0.coords[x]) )
        xdims.append(x)
        xcords[x]=xarr0.coords[x]
variables ={"nbr2": xr.DataArray(period_nbr2, dims=xdims,coords=ncoords)}
output=xr.Dataset(variables, attrs={'crs':xarr0.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=xarr0.coords[x].units
