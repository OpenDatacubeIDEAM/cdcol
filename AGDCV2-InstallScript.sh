#!/bin/bash
#ENG: Datacube install script on ubuntu 16.04 64bits
#ESP: Script para la instalación del cubo en una máquina virtual con Ubuntu Linux 16.04 64bits.
#Author: Christian Ariza - http://christian-ariza.net - mailto://cf.ariza975@uniandes.edu.co 

USUARIO_CUBO="$(whoami)"
#DEFAUTLS; 
PASSWORD_CUBO='ASDFADFASSDFA'
ANACONDA_URL="https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh"
REPO="https://github.com/cronosnull/agdc-v2.git"
BRANCH="develop"
#Options:
while getopts "a:p:r:b:h" opt; do
  case $opt in
    a) ANACONDA_URL="$OPTARG";;
    p) PASSWORD_CUBO="$OPTARG";;
	r) REPO="$OPTARG";;
	b) BRANCH="$OPTARG";;
	h) cat <<EOF
All arguments are optional
-a anaconda url
-p password postgress
-r repository 
-b branch
-h this help
EOF
exit 0;
;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done
echo "Installation will be performed as  $USUARIO_CUBO"

if [[ $(id -u) -eq 0 ]] ; then echo "This script must  not be excecuted as root or using sudo(althougth the user must be sudoer and password will be asked in some steps)" ; exit 1 ; fi
#Prerequisites installation: 
while fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
   echo "Waiting while other process ends installs (dpkg/lock is locked)"
   sleep 1
done
sudo apt install -y openssh-server postgresql-9.5 postgresql-client-9.5 postgresql-contrib-9.5 libgdal1-dev libhdf5-serial-dev libnetcdf-dev hdf5-tools netcdf-bin gdal-bin pgadmin3 postgresql-doc-9.5 libhdf5-doc netcdf-doc libgdal-doc git wget htop rabbitmq-server imagemagick ffmpeg nginx|| exit 1

if ! hash "conda" > /dev/null; then
	mkdir -p ~/instaladores && wget -c -P ~/instaladores $ANACONDA_URL
	bash ~/instaladores/Anaconda2-4.1.1-Linux-x86_64.sh -b -p $HOME/anaconda2
	export PATH="$HOME/anaconda2/bin:$PATH"
	echo 'export PATH="$HOME/anaconda2/bin:$PATH"'>>$HOME/.bashrc
fi

conda install -y psycopg2 gdal libgdal hdf5 rasterio netcdf4 libnetcdf pandas shapely ipywidgets scipy numpy

#AGDC Install: 
git clone $REPO
cd agdc-v2
git checkout $BRANCH
python setup.py install

#Database configuration:

sudo -u postgres psql postgres<<EOF
create user $USUARIO_CUBO with password '$PASSWORD_CUBO';
alter user $USUARIO_CUBO createdb;
alter user $USUARIO_CUBO createrole;
alter user $USUARIO_CUBO superuser;
EOF

createdb datacube

cat <<EOF >~/.datacube.conf
[datacube]
db_database: datacube

# A blank host will use a local socket. Specify a hostname to use TCP.
db_hostname: localhost

# Credentials are optional: you might have other Postgres authentication configured.
# The default username otherwise is the current user id.
db_username: $USUARIO_CUBO
db_password: $PASSWORD_CUBO
EOF

datacube -v system init
source $HOME/.bashrc
sudo groupadd ingesters
sudo mkdir /dc_storage
sudo mkdir /source_storage
sudo chown $USUARIO_CUBO:ingesters /dc_storage
sudo chmod -R g+rwxs /dc_storage
sudo chown $USUARIO_CUBO:ingesters /source_storage
sudo chmod -R g+rwxs /source_storage
sudo mkdir /web_storage
sudo chown $USUARIO_CUBO /web_storage
#Crear un usuario ingestor
pass=$(perl -e 'print crypt($ARGV[0], "password")' "uniandes")
sudo useradd  --no-create-home -G ingesters -p $pass ingestor --shell="/usr/sbin/nologin" --home /source_storage  -K UMASK=002

#TODO: At this point an empty datacube is installed. Next steps are create datasets types, index datasets and ingest.  
datacube product add ~/agdc-v2/docs/config_samples/dataset_types/ls7_scenes.yaml
datacube product add ~/agdc-v2/docs/config_samples/dataset_types/ls5_scenes.yaml
datacube product add ~/agdc-v2/docs/config_samples/dataset_types/ls8_scenes.yaml
datacube product add ~/agdc-v2/docs/config_samples/dataset_types/modis_tiles.yaml

#Celery Install
conda install -c conda-forge celery=3.1.23 flower 
sudo rabbitmqctl add_user cdcol cdcol
sudo rabbitmqctl add_vhost cdcol
sudo rabbitmqctl set_user_tags cdcol cdcol_tag
sudo rabbitmqctl set_permissions -p cdcol cdcol ".*" ".*" ".*"
sudo rabbitmq-plugins enable rabbitmq_management
sudo rabbitmqctl set_user_tags cdcol cdcol_tag administrator
sudo service rabbitmq-server restart
ip=`hostname -I | awk '{ print $1 }'`
echo "iniciando celery en la ip $ip en el puerto 8082"
nohup celery -A cdcol_celery worker --loglevel=info &
nohup celery -A cdcol_celery flower --port=8082 --address=$ip --persistent &

echo "Para que funcione el sftp del usuario ingestor se necesita cambiar 'Subsystem sftp /usr/lib/openssh/sftp-server' por 'Subsystem sftp internal-sftp' en /etc/ssh/sshd_config"

#Configuración de NFS
sudo apt-get install nfs-kernel-server
echo "¿Cuál es la ip del servidor web?"
read $ipweb
sudo sh -c 'echo "/dc_storage    $ipweb(ro,sync,no_subtree_check)">> /etc/exports'
sudo sh -c 'echo "/web_storage    $ipweb(rw,sync,no_subtree_check)">> /etc/exports'

sudo service nfs-kernel-server restart

#Instalación de los pre-requisitos del servicio web
conda install Django  simplejson pyyaml gunicorn 
conda install -c conda-forge djangorestframework

echo "Recuerde configurar GUnicorn y Nginx para exponer el servicio,
 cambiando el interprete para que sea el python de conda"