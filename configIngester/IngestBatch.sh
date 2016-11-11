#!/bin/bash
sudo renice -10 $$
nimg=0
basePath="${1:-.}" 
configFile="${2:-$HOME/configIngester/ls7_ledaps_wgs84.yaml}"
cd $basePath
basePath=$(pwd)
echo "Carpeta $basePath"
read -p "Press [Enter] key to continue..."
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
python ~/agdc-v2/utils/usgslsprepare.py $folder && datacube dataset add -a $folder
((nimg++))
if(( $nimg % 18 == 0 ))
then 
echo "Ingestion will be performed using $threads threads"
datacube ingest --executor multiproc $threads -c $configFile
rm -rf "/source_storage/tmp/"
fi
done
done

echo "Ingestion will be performed using $threads threads"
datacube ingest --executor multiproc $threads -c $configFile
rm -rf "/source_storage/tmp/"