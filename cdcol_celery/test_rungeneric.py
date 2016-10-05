from __future__ import absolute_import
from .tasks import longtime_add, medianas,generic_task
from celery import group
import time

if __name__ == '__main__':
#    result = longtime_add.delay(1,2)
 #   print result
    # at this time, our task is not finished, so it will return False
#    print 'Task finished? ', result.ready()
 #   print 'Task result: ', result.result
    # sleep 10 seconds to ensure the task has been finished
#    time.sleep(10)
    # now the task should be finished and ready method will return True
#    print 'Task finished? ', result.ready()
#    print 'Task result: ', result.result
    result = group(generic_task.s(execID="122112",algorithm="medianas",version="1",output_expression="",product="ls7_ledaps_utm18n4",min_lat=Y,min_long=X,time_ranges=[("2000-01-01","2000-12-31")],normalized=True,bands=["blue","green","red","nir", "swir1","swir2"], minValid=1) for Y in xrange(5,6) for X in xrange(-75,-74)).delay()
    print result
    print result.ready()
