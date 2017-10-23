from __future__ import absolute_import
from .tasks import generic_task_mp
from celery import group
import time

if __name__ == '__main__':
    #Test medianas
    result = group(generic_task_mp.s(execID="medianas2_mp",algorithm="medianas",version="2.0",output_expression="",products=["LS7_ETM_LEDAPS","LS8_OLI_LEDAPS"],min_lat=Y,min_long=X,time_ranges=[("2013-01-01","2014-12-31")],normalized=True,bands=["blue","green","red","nir", "swir1","swir2"], minValid=1) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    print result
    #Test ndvi 
    #result = group(generic_task_mp.s(execID="ndvi1_mp",algorithm="ndvi",version="1.0",output_expression="",products=["LS7_ETM_LEDAPS","LS8_OLI_LEDAPS"],min_lat=Y,min_long=X,time_ranges=[("2014-01-01","2014-12-31")],normalized=True, minValid=1) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    #print result
    #Test forest-noforest
    #result = group(generic_task_mp.s(execID="forest1_mp",algorithm="forest_noforest",version="1.0",output_expression="",products=["LS7_ETM_LEDAPS","LS8_OLI_LEDAPS"],min_lat=Y,min_long=X,time_ranges=[("2014-01-01","2014-12-30")],normalized=True, minValid=1, vegetation_rate = 0.7,ndvi_threshold = 0.3, slice_size = 3, ok_pixels = 1) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    #print result
    #Test PCA
    #result = group(generic_task_mp.s(execID="pca_mp",algorithm="pca",version="1.0",output_expression="",products=["LS7_ETM_LEDAPS","LS8_OLI_LEDAPS"],min_lat=Y,min_long=X,time_ranges=[("2014-01-01","2014-06-30"),("2014-07-01","2014-12-31")],normalized=True, minValid=1,bands=["blue","green","red","nir", "swir1","swir2"] ) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    #print result 
    #Test WOFS
    #result = group(generic_task_mp.s(execID="wofs-exec2_mp",algorithm="wofs",version="1.0",output_expression="",products=["LS7_ETM_LEDAPS","LS8_OLI_LEDAPS"],min_lat=Y,min_long=X,time_ranges=[("2014-01-01","2014-12-31")]) for Y in xrange(1,2) for X in xrange(-74,-73)).delay()
    #print result

    

