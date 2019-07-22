# Manual Ingestion Scrits 

The ingestion script process an ingestion task by blocks of 18 images. 

* The images are unziped.
* Indexed and Ingested.
* The script takes the next block of 18 images.

A satellite image comes in a .tar.gz like ```LC080060582017121301T1-SC20190705222131.tar.gz```. This file is unzipped in the folder ```LC080060582017121301T1```. The folder content includes some .tiff files which represent the image bands (or spectral response).

```
LC08_L1TP_006058_20171213_20171223_01_T1_ANG.txt
LC08_L1TP_006058_20171213_20171223_01_T1_MTL.txt
LC08_L1TP_006058_20171213_20171223_01_T1_pixel_qa.tif
LC08_L1TP_006058_20171213_20171223_01_T1_radsat_qa.tif
LC08_L1TP_006058_20171213_20171223_01_T1_sr_aerosol.tif
LC08_L1TP_006058_20171213_20171223_01_T1_sr_band1.tif
LC08_L1TP_006058_20171213_20171223_01_T1_sr_band2.tif
LC08_L1TP_006058_20171213_20171223_01_T1_sr_band3.tif
LC08_L1TP_006058_20171213_20171223_01_T1_sr_band4.tif
LC08_L1TP_006058_20171213_20171223_01_T1_sr_band5.tif
LC08_L1TP_006058_20171213_20171223_01_T1_sr_band6.tif
LC08_L1TP_006058_20171213_20171223_01_T1_sr_band7.tif
LC08_L1TP_006058_20171213_20171223_01_T1.xml
```

## Performing an ingestion

Ingesting LS5_TM_LEDAPS

```sh
nohup ./IngestBatch.sh /source_storage/LS5_TM_LEDAPS/ /dc_storage/LS5_TM_LEDAPS/ingest_file.yml
```

Ingesting LS7_ETM_LEDAPS

Landzat 7 comes with ToA Reflectance bands. This images must not be indexed or ingested in the current 
datacube deployment so this script removes that files for LS7 ingestion.

```sh
nohup ./IngestBatch.sh /source_storage/LS7_ETM_LEDAPS/ /dc_storage/LS7_ETM_LEDAPS/ingest_file.yml /dc_storage/LS7_ETM_LEDAPS/mgen_script.py
```

Ingesting LS8_OLI_LASRC

```sh
nohup ./IngestBatch.sh /source_storage/LS8_OLI_LASRC/ /dc_storage/LS8_OLI_LASRC/ingest_file.yml /dc_storage/LS8_OLI_LASRC/mgen_script.py
```

