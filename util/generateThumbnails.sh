#!/bin/bash
#Parámetros (opcionales) Unidad de almacenamiento, resolución. 
STORAGE_UNIT="${1:-LS7_ETM_LEDAPS}" 
TN_FOLDER=/web_storage/thumbnails
RES="${2:-500}"
mkdir -p $TN_FOLDER/$STORAGE_UNIT
for file in /dc_storage/$STORAGE_UNIT/*.nc
do
for ds in `gdalinfo $file |grep -Eo "NETCDF.*"`
do
bn=`basename $file`
echo "Escribiendo los thumbnails para el archivo $file y la banda ${ds##*\:}"
gdal_translate  -a_srs EPSG:32618 -a_nodata -9999 -stats -of PNG -scale -outsize $RES $RES $ds $TN_FOLDER/$STORAGE_UNIT/${bn%.nc}.${ds##*\:}.png
done
done