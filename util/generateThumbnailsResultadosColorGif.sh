#!/bin/bash
#Script para la generación de thumbnails de resultados, 
#debería ser llamado por el cron que revisa el estado de la ejecución, 
#cuando encuentra que una ejecución ha terminado correctamente.
#Parámetros Opcionales:Carpeta resolución

#GDAL_DATA Debería estar definida en el entorno, si no, toca definirla.
export GDAL_DATA="${GDAL_DATA:-/usr/share/gdal/1.11}"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $GDAL_DATA
FOLDER="${1:-.}"
TN_FOLDER="$FOLDER/thumbnails"
RES="${2:-500}"
mkdir -p $TN_FOLDER
for file in $FOLDER/*.nc
do
bn=`basename $file`
if `gdalinfo $file |grep -q "SUBDATASET.*"`
then
	for ds in `gdalinfo $file |grep -Eo "NETCDF.*"`
	do
	echo "Escribiendo los thumbnails para el archivo $file y la banda ${ds##*\:}"
	gdal_translate  -a_srs EPSG:32618   -stats -of PNG -scale -ot Byte -outsize $RES $RES $ds $TN_FOLDER/${bn%.nc}.${ds##*\:}.png
	convert -transparent "#000000" $TN_FOLDER/${bn%.nc}.${ds##*\:}.png $DIR/colores/rainbow1.png -clut $TN_FOLDER/${bn%.nc}.${ds##*\:}.png
	done
else
	
	nb=`gdalinfo $file |grep  Band|wc -l`
	echo $nb 
	if [[ $nb -le 1 ]]
	then
		echo "Escribiendo el thumbnail para el archivo $file"
		gdal_translate  -a_srs EPSG:32618  -stats -of PNG -scale -ot Byte -outsize $RES $RES $file $TN_FOLDER/${bn%.nc}.png
		convert -transparent "#000000" $TN_FOLDER/${bn%.nc}.png $DIR/colores/rainbow1.png -clut $TN_FOLDER/${bn%.nc}.png
	else
		for i in $(seq 1 $nb)
		do
		echo "Escribiendo el thumbnail para el archivo $file banda $i"

		gdal_translate  -a_srs EPSG:32618  -stats -of PNG -scale -ot Byte -b $i -outsize $RES $RES $file $TN_FOLDER/${bn%.nc}.$i.png
		convert -transparent "#000000" $TN_FOLDER/${bn%.nc}.$i.png $DIR/colores/rainbow1.png -clut $TN_FOLDER/${bn%.nc}.$i.png
		done
	fi
fi
done

convert -dispose previous -delay 20 -loop 0 -coalesce   $TN_FOLDER/*.png $TN_FOLDER/animated.gif