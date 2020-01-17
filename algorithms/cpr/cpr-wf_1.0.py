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

#CÁLCULO DE 2DO ÍNDICE SPR
s1  = (banda1+banda2)
s2  =(banda1-banda2)
s3 = 2 * (banda1 **(1/2))*(banda2 **(1/2)) * np.cos(banda1-banda2)
s4 =2 * (banda1 **(1/2))*(banda2 **(1/2)) * np.sin(banda1-banda2)
LPR =(s1+s2)/(s1-s2)
SC = 0.5*s1 - 0.5 * s4
OC = 0.5*s1 + 0.5 * s4
CPR =(SC/OC)
m = (((s2**2)+(s3**2)+(s4**2))**(1/2))/s1
period_cpr=[]

period_cpr['cpR']=CPR[0].values

print('Indice calculado')
del datos


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

print("cpr")
print(output)
