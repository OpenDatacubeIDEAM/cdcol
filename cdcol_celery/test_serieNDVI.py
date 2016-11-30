from __future__ import absolute_import
from .tasks import generic_task, runCmd
from celery import group,chord
from os.path import expanduser
import time
import sys
if __name__ == '__main__':
    exec_id="pruebaNDVI-1"
    min_lat=10
    min_long=-75
    anio_i=2014
    anio_f=2015 
    if(len(sys.argv)>2):
        exec_id=sys.argv[1]
    if(len(sys.argv)>3):
        anio_i=int(sys.argv[2])
        anio_f=int(sys.argv[3])
    if(len(sys.argv)>5):
        min_lat=int(sys.argv[4])
        min_long=int(sys.argv[5])
    result = group(generic_task.s(execID=exec_id,algorithm="ndvi",version="1.1",output_expression="",product="LS7_ETM_LEDAPS",min_lat=min_lat,min_long=min_long,time_ranges=[(str(A)+"-01-01",str(A)+"-12-31")],normalized=True, minValid=1) for A in xrange(anio_i,anio_f+1)).delay()
    print result
    result.get(propagate=False)
    runCmd(execID=exec_id, cmd=['bash',expanduser("~")+'/util/generateThumbnailsResultadosColorGif.sh','/Results/'+exec_id+'/'])