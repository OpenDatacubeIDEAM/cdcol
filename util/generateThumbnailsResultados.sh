#!/bin/bash
#Script para la generación de thumbnails de resultados, 
#debería ser llamado por el cron que revisa el estado de la ejecución, 
#cuando encuentra que una ejecución ha terminado correctamente.
#Parámetros Opcionales:Carpeta resolución

#GDAL_DATA Debería estar definida en el entorno, si no, toca definirla.
GDAL_DATA="${GDAL_DATA:-/usr/share/gdal/1.11}"

FOLDER="${1:-.}"
TN_FOLDER="$FOLDER/thumbnails"
RES="${2:-500}"
mkdir -p $TN_FOLDER
for file in $FOLDER/*.nc
do
for ds in `gdalinfo $file |grep -Eo "NETCDF.*"`
do
bn=`basename $file`
if `gdalinfo $file |grep -q "SUBDATASET.*"`
then
echo "Escribiendo los thumbnails para el archivo $file y la banda ${ds##*\:}"
gdal_translate  -a_srs EPSG:32618 -a_nodata -9999 -stats -of PNG -scale -outsize $RES $RES $ds $TN_FOLDER/${bn%.nc}.${ds##*\:}.png
else
echo "Escribiendo el thumbnail para el archivo $file"
gdal_translate  -a_srs EPSG:32618 -a_nodata -9999 -stats -of PNG -scale -outsize $RES $RES $file $TN_FOLDER/${bn%.nc}.png
fi
done
done
