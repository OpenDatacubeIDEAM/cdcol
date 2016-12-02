#!/bin/bash
#Parámetros (opcionales) Unidad de almacenamiento, resolución. 
export GDAL_DATA="${GDAL_DATA:-/usr/share/gdal/1.11}"
STORAGE_UNIT="${1:-LS7_ETM_LEDAPS}" 
TN_FOLDER=/web_storage/thumbnails
RES="${2:-500}"
CLU='panoply'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p $TN_FOLDER/$STORAGE_UNIT
for file in /dc_storage/$STORAGE_UNIT/*.nc
do
for ds in `gdalinfo $file |grep -Eo "NETCDF.*"`
do
bn=`basename $file`
echo "Escribiendo los thumbnails para el archivo $file y la banda ${ds##*\:}"
OFPNG=$TN_FOLDER/$STORAGE_UNIT/${bn%.nc}.${ds##*\:}.png
(gdal_translate -q  -a_srs EPSG:32618 -stats -of PNG -scale -ot Byte -outsize $RES $RES $ds $OFPNG  2>~/errors|| (rm $OFPNG && false)) && \
convert -transparent "#000000" $OFPNG $DIR/colores/$CLU.png -clut $OFPNG &&\
convert $OFPNG   -background Khaki  label:"${bn%.nc}.${ds##*\:}" -gravity Center -append  $OFPNG &
done
done