import xarray as xr
import numpy as np
print("Bosque no bosque ")

nodata=-9999
period_nvdi=xarr0["ndvi"].values
height = period_nvdi.shape[0]
width = period_nvdi.shape[1]
#bosque_nobosque=np.full(period_nvdi.shape, -1, dtype='int16' )
bosque_nobosque=np.full(period_nvdi.shape, -1, dtype='int32' )
for y1 in range(0, height, slice_size):
    for x1 in range(0, width, slice_size):
        x2 = x1 + slice_size
        y2 = y1 + slice_size
        if(x2 > width):
            x2 = width
        if(y2 > height):
            y2 = height
        submatrix = period_nvdi[y1:y2,x1:x2]
        ok_pixels = np.count_nonzero(~np.isnan(submatrix))
        if ok_pixels==0:
            bosque_nobosque[y1:y2,x1:x2] = nodata
        elif np.nansum(submatrix>ndvi_threshold)/ok_pixels >= vegetation_rate :
            bosque_nobosque[y1:y2,x1:x2] = 1
        else:
            bosque_nobosque[y1:y2,x1:x2] = 0


ncoords=[]
xdims =[]
xcords={}
for x in xarr0.coords:
    if(x!='time'):
        ncoords.append( ( x, xarr0.coords[x]) )
        xdims.append(x)
        xcords[x]=xarr0.coords[x]

#variables ={"bosque_nobosque": xr.DataArray(bosque_nobosque, dims=xdims,coords=ncoords)}
array = xr.DataArray(bosque_nobosque, dims=xdims,coords=ncoords)
array = array.astype('float32')
variables ={"bosque_nobosque": array}

output=xr.Dataset(variables, attrs={'crs':xarr0.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=xarr0.coords[x].units
