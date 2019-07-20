#!/bin/bash
sudo renice -10 $$
nimg=0
basePath="${1:-.}" 
configFile="${2:-$HOME/configIngester/ls7_ledaps_wgs84.yaml}"
cd $basePath
basePath=$(pwd)
echo "Carpeta $basePath"
read -p "Press [Enter] key to continue..."
export GDAL_DATA="${GDAL_DATA:-/usr/share/gdal/1.11}"
threads=$( expr $(grep -c ^processor /proc/cpuinfo) - 1)
if [ $threads -eq 0 ] 
then
	$threads = 1
fi
for dirE in $basePath/*/
do
echo $dirE
cd $dirE
for archivo in *.tar.gz
do
echo ${archivo%%-*}
#read -p "Press [Enter] key to continue..."
folder="/source_storage/tmp/${archivo%%-*}"
#En teoría pigz tiene mejor rendimiento que gzip para descomprimir. 
mkdir -p $folder  && tar -xzf $archivo -C $folder
rm $folder/*toa*
python ~/agdc-v2/utils/usgslsprepare.py $folder && datacube dataset add -a $folder
((nimg++))
done
done
