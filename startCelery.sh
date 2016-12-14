ip=`hostname -I | awk '{ print $1 }'`
echo "iniciando celery en la ip $ip en el puerto 8082"
nohup celery -A cdcol_celery worker --loglevel=info &
nohup celery -A cdcol_celery flower --port=8082 --address=$ip --persistent &