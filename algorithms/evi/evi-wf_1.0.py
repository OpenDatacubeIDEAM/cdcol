import xarray as xr
import numpy as np
print("Indice de vegetacion mejorado")

period_red = xarr0["red"].values
period_nir = xarr0["nir"].values
period_blue = xarr0["blue"].values

mask_nan=np.logical_or(np.isnan(period_red), np.isnan(period_nir),np.isnan(period_blue))
a= (period_nir-period_red)
b= (period_nir+period_red-(7.5*period_blue)+1)
period_evi= 2.5*(a/b)

period_evi[mask_nan]=np.nan
#Hace un clip para evitar valores extremos. 
period_evi[period_evi>1]=np.nan
period_evi[period_evi<-1]=np.nan


ncoords=[]
xdims =[]
xcords={}
for x in xarr0.coords:
    if(x!='time'):
        ncoords.append( ( x, xarr0.coords[x]) )
        xdims.append(x)
        xcords[x]=xarr0.coords[x]
variables ={"evi": xr.DataArray(period_evi, dims=xdims,coords=ncoords)}
output=xr.Dataset(variables, attrs={'crs':xarr0.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=xarr0.coords[x].units

print("evi")
print(output)
