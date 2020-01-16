#!/usr/bin/python3
# coding=utf8
import xarray as xr
import numpy as np
print ("Compuesto temporal de medianas para " + product['name'])
print(xarr0)
nodata=-9999
medians = {}
time_axis = list(xarr0.coords.keys()).index('time')
for band in product['bands']:
    if band != 'pixel_qa':
        datos = xarr0.data_vars[band].values
        allNan = ~np.isnan(datos)

        # Comentada por Aurelio (No soporta multi unidad)
        #if normalized:
        #    m=np.nanmean(datos.reshape((datos.shape[time_axis],-1)), axis=1)
        #    st=np.nanstd(datos.reshape((datos.shape[time_axis],-1)), axis=1)
        #    datos=np.true_divide((datos-m[:,np.newaxis,np.newaxis]), st[:,np.newaxis,np.newaxis])*np.nanmean(st)+np.nanmean(m)

        if normalized:
            m=np.nanmean(datos.reshape((datos.shape[time_axis],-1)), axis=1)
            st=np.nanstd(datos.reshape((datos.shape[time_axis],-1)), axis=1)

            # Expand m and st according with the data shape
            # number of coords
            coords_num = len(list(xarr0.coords.keys()))
            l = [ x for x in range(coords_num) if x != time_axis]

            m_new = m
            st_new = st
            for axis in l:
                # If axis is 0  it is equivalent to x[np.newaxis,:]
                # If axis is 1  it is equivalent to x[:,np.newaxis]
                # And so on
                m_new = np.expand_dims(m_new,axis=axis)
                st_new = np.expand_dims(st_new,axis=axis)

            print('Time axis',time_axis)
            print('New axis',l)
            print('m',m.shape)
            print('st',st.shape)
            print('st_new',st_new.shape)
            print('m_new',m_new.shape)
            datos=np.true_divide((datos-m_new), st_new)*np.nanmean(st)+np.nanmean(m)

        medians[band] = np.nanmedian(datos, time_axis)
        medians[band][np.sum(allNan, time_axis) < minValid] = -9999
        del datos

# > **Asignación de coordenadas**
ncoords=[]
xdims =[]
xcords={}
for x in xarr0.coords:
    if(x!='time'):
        ncoords.append( ( x, xarr0.coords[x]) )
        xdims.append(x)
        xcords[x]=xarr0.coords[x]
variables ={k: xr.DataArray(v, dims=xdims,coords=ncoords) for k, v in medians.items()}
output=xr.Dataset(variables, attrs={'crs':xarr0.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=xarr0.coords[x].units

