from __future__ import absolute_import
from .tasks import generic_task, runCmd
from celery import group,chord
from os.path import expanduser
import time
if __name__ == '__main__':
    exec_id="pruebaNDVI"
    min_lat=0
    min_long=-74
    anio_i=2014
    anio_f=2015
    result = group(generic_task.s(execID=exec_id,algorithm="ndvi",version="1.1",output_expression="",product="LS8_OLI_LEDAPS",min_lat=min_lat,min_long=min_long,time_ranges=[(str(A)+"-01-01",str(A)+"-12-31")],normalized=True, minValid=1) for A in xrange(anio_i,anio_f+1)).delay()
    print result
    result.get()
    runCmd(execID=exec_id, cmd=['bash',expanduser("~")+'/util/generateThumbnailsResultadosGif.sh','/Results/'+exec_id+'/'])