#!/home/developer/anaconda2/bin/python
# coding=utf8
from __future__ import absolute_import
from celery import Celery

app = Celery('cdcol_celery',
             broker='amqp://cdcol:cdcol@localhost/cdcol',
             backend='rpc://',
             include=['cdcol_celery.tasks'])
app.conf.CELERYD_PREFETCH_MULTIPLIER = 1
app.conf.CELERYD_MAX_TASKS_PER_CHILD = 1

##Si se desea limitar el número de workers, se puede usar CELERYD_CONCURRENCY.
##Debe ser menor al número de cores disponibles. 
##(por defecto toma el número de cores)
app.conf.CELERYD_CONCURRENCY = 3 #Por ejemplo, en este caso quiero sólo 3 workers. 