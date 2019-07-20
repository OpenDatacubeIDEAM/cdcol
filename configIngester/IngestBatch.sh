#!/bin/bash

sudo renice -10 $$

# Script base directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


baseIngestPath="${1}" 
ingestConfigFile="${2}"
matadataPrepareScript="${3:-$DIR/metadata_prepare_scripts/agdc-v2/usgslsprepare.py}"

echo "-------------------------"
echo "This script was designed for ingestion in LS5, LS7 and LS8."
echo "In case of LS7 removes the toa file comming in LS7 image zip\
 files that must not be ingested in the Colombian datacube."

echo "=> Date: $(date)"
echo "=> Data to ingest:                    $baseIngestPath"
echo "=> Ingest Config File (.yml):         $ingestConfigFile"
echo "=> Metadata generation Script (.py):  $matadataPrepareScript"

read -p "Press [Enter] key to continue..."

threads=$( expr $(grep -c ^processor /proc/cpuinfo) - 1)
if [ $threads -eq 0 ] ; then $threads = 1; fi

export GDAL_DATA="${GDAL_DATA:-/usr/share/gdal/1.11}"
nimg=0

echo "=> Entry into the $baseIngestPath"
cd $baseIngestPath

for archivo in *.tar.gz
do
	echo "=> $archivo found."
	folder="/source_storage/tmp/${archivo%%-*}"
	 
	echo "=> Unzip $archivo into $folder"
	mkdir -p $folder tar -xzf $archivo -C $folder

	echo "=> Remove toa files from $folder"
	rm -v -f $folder/*toa*

	echo "=> Create image metadata and performing datacube add in $folder folder"
	python $matadataPrepareScript $folder && datacube dataset add -a $folder

	((nimg++))
	if(( $nimg % 18 == 0 ))
	then 
		echo "=> Ingestion will be performed using $threads threads"
		datacube -v ingest --allow-product-changes --executor multiproc $threads -c $ingestConfigFile
		rm -rf "/source_storage/tmp/"
	fi
done


echo "Ingestion will be performed using $threads threads"
datacube -v ingest --allow-product-changes --executor multiproc $threads -c $ingestConfigFile
rm -rf "/source_storage/tmp/"
