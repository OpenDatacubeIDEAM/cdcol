#!/usr/bin/python3
# coding=utf8
import xarray as xr
import numpy as np
print(xarr0)
nodata=-9999
#medians = {}
time_axis = list(xarr0.coords.keys()).index('time')

print(xarr0)
print(type(xarr0))

list_bandas=list(xarr0.data_vars)
print('bandas anterior codigo')

banda1=xarr0['hh']
banda2=xarr0['hv']

#Cálculo del 1er RDFI índice Radar Forest Degradation  para ALOS PALSAR
RDFI =(banda1-banda2)/(banda1+banda2)
print('Indice calculado')


ncoords=[]
xdims =[]
xcords={}
for x in xarr0.coords:
    if(x!='time'):
        ncoords.append( ( x, xarr0.coords[x]) )
        xdims.append(x)
        xcords[x]=xarr0.coords[x]
variables ={"cpr": xr.DataArray(period_cpr, dims=xdims,coords=ncoords)}
output=xr.Dataset(variables, attrs={'crs':xarr0.crs})
for x in output.coords:
    output.coords[x].attrs["units"]=xarr0.coords[x].units

print("RDFI")
print(output)
