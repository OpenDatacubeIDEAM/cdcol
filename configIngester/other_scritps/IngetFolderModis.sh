#!/bin/bash
#Script para ingestar un conjunto de imágenes USGS (modis43a4) realizando los procesos previos (preparar, indexar) - cada dataset modis está en una subcarpeta, sin comprimir. 
#El primer parámetro es la carpeta en la que se encuentran las carpetas con los datasets modis del USGS (modis43a4), si se llama sin parámetros toma la carpeta actual
#El segundo parámetro es el archivo de configuración de ingesta, por defecto usará el de modis43a4 utm18n
basePath="${1:-.}" 
configFile="${2:-~/configIngester/mcd43a4_utm18n.yaml}"
cd $basePath
for folder in */
do
python ~/agdc-v2/utils/modisprepare.py $folder && datacube dataset add -a $folder
done
threads=$( expr $(grep -c ^processor /proc/cpuinfo) - 1)
if [ $threads -eq 0 ] 
then
	$threads = 1
fi
echo "Ingestion will be performed using $threads threads"
datacube ingest --executor multiproc $threads -c $configFile
rm -rf tmp
