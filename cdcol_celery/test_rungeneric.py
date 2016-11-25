from __future__ import absolute_import
from .tasks import generic_task
from celery import group
import time

if __name__ == '__main__':
    #Test medianas
    result = group(generic_task.s(execID="medianas2",algorithm="medianas",version="2",output_expression="",product="LS8_OLI_LEDAPS",min_lat=Y,min_long=X,time_ranges=[("2014-01-01","2014-06-30")],normalized=True,bands=["blue","green","red","nir", "swir1","swir2"], minValid=1) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    print result
    #Test ndvi 
    result = group(generic_task.s(execID="ndvi1",algorithm="ndvi",version="1",output_expression="",product="LS8_OLI_LEDAPS",min_lat=Y,min_long=X,time_ranges=[("2014-01-01","2014-06-30")],normalized=True, minValid=1) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    print result
    #Test forest-noforest
    result = group(generic_task.s(execID="forest1",algorithm="forest_noforest",version="1",output_expression="",product="LS8_OLI_LEDAPS",min_lat=Y,min_long=X,time_ranges=[("2014-01-01","2014-06-30")],normalized=True, minValid=1, vegetation_rate = 0.7,ndvi_threshold = 0.3, slice_size = 3, ok_pixels = 1) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    print result
    #Test PCA
    result = group(generic_task.s(execID="pca",algorithm="pca",version="1",output_expression="",product="LS8_OLI_LEDAPS",min_lat=Y,min_long=X,time_ranges=[("2014-01-01","2014-06-30"),("2014-07-01","2014-12-31")],normalized=True, minValid=1,bands=["blue","green","red","nir", "swir1","swir2"] ) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    print result 
    #Test WOFS
    result = group(generic_task.s(execID="wofs-exec2",algorithm="wofs",version="1.0",output_expression="",product="LS8_OLI_LEDAPS",min_lat=Y,min_long=X,time_ranges=[("2014-01-01","2014-12-31")]) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    print result

    

