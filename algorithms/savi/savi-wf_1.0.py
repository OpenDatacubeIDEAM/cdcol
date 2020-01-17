import xarray as xr
import numpy as np
print("Indice de Vegetacion")


period_red = xarr0["red"].values
period_nir = xarr0["nir"].values

print("mascara")
mask_nan=np.logical_or(np.isnan(period_nir), np.isnan(period_red))
period_savi1 = (period_nir-period_red)/(period_nir+period_red+0.5) 
period_savi = (1+0.5)*period_savi1
period_savi[mask_nan]=np.nan
#Hace un clip para evitar valores extremos.
period_nbr[period_savi>1]=np.nan
period_nbr[period_savi<-1]=np.nan

ncoords=[]
xdims =[]
xcords={}
for x in xarr0.coords:
    if(x!='time'):
        ncoords.append( ( x, xarr0.coords[x]) )
        xdims.append(x)
        xcords[x]=xarr0.coords[x]
variables ={"nbr": xr.DataArray(period_savi
, dims=xdims,coords=ncoords)}
output=xr.Dataset(variables, attrs={'crs':xarr0.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=xarr0.coords[x].units
