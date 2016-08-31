#!/bin/bash
#Script para ingestar un conjunto de imágenes USGS (landsat 7 ledaps) realizando los procesos previos (descomprimir, preparar, indexar)
#El primer parámetro es la carpeta en la que se encuentran los tar.gz del USGS, si se llama sin parámetros toma la carpeta actual
basePath="${1:-.}" 
cd basePath
for archivo in *.tar.gz
do
folder="tmp/${archivo%%.*}"
base=${archivo%%-*}
mkdir -p $folder  && tar -xzf $archivo -C $folder
python ~/agdc-v2/utils/usgslsprepare.py $folder && datacube dataset add -a $folder
done
datacube ingest  -c ~/configIngester/ls7_ledaps_wgs84.yaml
rm -rf tmp