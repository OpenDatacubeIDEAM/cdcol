#!/usr/bin/env python
# -*- coding: utf-8 -*-

execID=17
algorithm = "MosaicoDosUnidades"
version= "1.0"



products = ['LS8_OLI_LASRC','LS7_ETM_LEDAPS' ]
bands=["blue","green","red","nir", "swir1","swir2"]
time_ranges = [("2016-01-01", "2016-12-31")]

min_long = -79
min_lat = -5
max_long = -66
max_lat = 13


normalized=True
minValid=1;


import datacube
from datacube.storage import netcdf_writer
from datacube.model import Variable, CRS
import os
import re
import xarray as xr
import numpy as np
import gdal
import itertools


nodata=-9999

def isin(element, test_elements, assume_unique=False, invert=False):
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)





dc = datacube.Datacube(app="{}_{}_{}".format(algorithm,version,execID))

for _lat, _lon in itertools.product(range(min_lat, max_lat), range(min_long, max_long, 1)):
    kwargs={}
    for product in products:
        i=0
        validValues=set()
        if product=="LS7_ETM_LEDAPS":
            validValues=[66,68,130,132]
        elif product == "LS8_OLI_LASRC":
            validValues=[322, 386, 834, 898, 1346, 324, 388, 836, 900, 1348]
        for tr in time_ranges:
            _data = dc.load(product=product, longitude=(_lon, _lon+1), latitude=(_lat, _lat+1), time=tr)
            if len(_data.data_vars)==0:
                break
            cloud_mask=isin(_data["pixel_qa"].values, validValues)
            for band in bands:
                _data[band].values=np.where(np.logical_and(_data.data_vars[band]!=nodata,cloud_mask),_data.data_vars[band], np.nan)
            _undesired=list(set(_data.keys())-set(bands+['latitude','longitude','time']))
            _data=_data.drop(_undesired)

            if "xarr"+str(i) in kwargs:
                kwargs["xarr"+str(i)]=xr.concat([kwargs["xarr"+str(i)],_data.copy(deep=True)], 'time')
            else:
                kwargs["xarr"+str(i)]=_data
        i+=1
    del _data
    if not "xarr0" in kwargs:
        continue
    xarr0=kwargs["xarr0"]
    del kwargs
    if len(xarr0.data_vars) == 0:
        continue
    medians={}
    for band in bands:
        datos=xarr0[band].values
        allNan=~np.isnan(datos)
        if normalized:

            m=np.nanmean(datos.reshape((datos.shape[0],-1)), axis=1)
            st=np.nanstd(datos.reshape((datos.shape[0],-1)), axis=1)


            datos=np.true_divide((datos-m[:,np.newaxis,np.newaxis]), st[:,np.newaxis,np.newaxis])*np.nanmean(st)+np.nanmean(m)

        medians[band]=np.nanmedian(datos,0)

        medians[band][np.sum(allNan,0)<minValid]=np.nan
    del datos


    _coords=xarr0.coords
    _crs=xarr0.crs
    del xarr0



    import xarray as xr
    ncoords=[]
    xdims =[]
    xcords={}
    for x in _coords:
        if(x!='time'):
            ncoords.append( ( x, _coords[x]) )
            xdims.append(x)
            xcords[x]=_coords[x]
    variables ={k: xr.DataArray(v, dims=xdims,coords=ncoords)
                 for k, v in medians.items()}
    output=xr.Dataset(variables, attrs={'crs':_crs})

    for x in output.coords:
        _coords[x].attrs["units"]=_coords[x].units


    from datacube.storage import netcdf_writer
    from datacube.model import Variable, CRS
    print "{}_{}_{}_{}_{}.nc".format(algorithm,version,execID,_lat,_lon)
    nco=netcdf_writer.create_netcdf("{}_{}_{}_{}_{}.nc".format(algorithm,version,execID,_lat,_lon))
    cords=('latitude', 'longitude','time')
    for x in cords:
        if(x!="time"):
            netcdf_writer.create_coordinate(nco, x, _coords[x].values, _coords[x].units)
    netcdf_writer.create_grid_mapping_variable(nco, _crs)
    for band in bands:
        medians[band][np.isnan(medians[band])]=nodata
        var= netcdf_writer.create_variable(nco, band, Variable(np.dtype(np.int32), None, ('latitude', 'longitude'), None) ,set_crs=True)
        var[:] = netcdf_writer.netcdfy_data(medians[band])
    nco.close()
    del nco
    del output
    del medians
