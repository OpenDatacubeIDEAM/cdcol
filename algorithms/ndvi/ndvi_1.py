import xarray as xr
import numpy as np
print "Excecuting ndvi v1 "
nbar = xarr0
nodata=-9999
bands=["red","nir"]
medians={}
cloud_mask=np.where(np.logical_and(nbar["cf_mask"].values!=2, nbar["cf_mask"].values<4), True, False)
for band in bands:
    datos=np.where(np.logical_and(nbar.data_vars[band]!=nodata,cloud_mask),nbar.data_vars[band], np.nan)
    allNan=~np.isnan(datos)
    if normalized:
        m=np.nanmean(datos.reshape((datos.shape[0],-1)), axis=1)
        st=np.nanstd(datos.reshape((datos.shape[0],-1)), axis=1)
        datos=np.true_divide((datos-m[:,np.newaxis,np.newaxis]), st[:,np.newaxis,np.newaxis])*np.nanmean(st)+np.nanmean(m)
    medians[band]=np.nanmedian(datos,0)
    medians[band][np.sum(allNan,0)<minValid]=np.nan
del datos
period_red = medians["red"]
period_nir = medians["nir"]
del medians
mask_nan=np.logical_or(np.isnan(period_red), np.isnan(period_nir))
period_nvdi = np.true_divide( np.subtract(period_nir,period_red) , np.add(period_nir,period_red) )
period_nvdi[mask_nan]=np.nan
import xarray as xr
ncoords=[]
xdims =[]
xcords={}
for x in nbar.coords:
    if(x!='time'):
        ncoords.append( ( x, nbar.coords[x]) )
        xdims.append(x)
        xcords[x]=nbar.coords[x]
variables ={"ndvi": xr.DataArray(period_nvdi, dims=xdims,coords=ncoords)}
output=xr.Dataset(variables, attrs={'crs':nbar.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=nbar.coords[x].units
print output