#!/bin/bash
#Script para ingestar un conjunto de imágenes USGS (landsat 7 ledaps) realizando los procesos previos (descomprimir, preparar, indexar)
#El primer parámetro es la carpeta en la que se encuentran los tar.gz del USGS, si se llama sin parámetros toma la carpeta actual
#El segundo parámetro es el archivo de configuración de ingesta, por defecto usará el de landsat 7 wgs84
basePath="${1:-.}" 
configFile="${2:-$HOME/configIngester/ls7_ledaps_wgs84.yaml}"
cd $basePath
for archivo in *.tar.gz
do
folder="tmp/${archivo%%.*}"
base=${archivo%%-*}
mkdir -p $folder  && tar -xzf $archivo -C $folder
python ~/agdc-v2/utils/usgslsprepare.py $folder && datacube dataset add -a $folder
done
threads=$( expr $(grep -c ^processor /proc/cpuinfo) - 1)
if [ $threads -eq 0 ] 
then
	$threads = 1
fi
echo "Ingestion will be performed using $threads threads"
datacube ingest --executor multiproc $threads -c $configFile
rm -rf tmp