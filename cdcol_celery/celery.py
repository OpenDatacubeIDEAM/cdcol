#!/home/developer/anaconda2/bin/python
from __future__ import absolute_import
from celery import Celery

app = Celery('cdcol_celery',
             broker='amqp://cdcol:cdcol@localhost/cdcol',
             backend='rpc://',
             include=['cdcol_celery.tasks'])