# coding: utf-8
from datacube.ui import click as ui
from datacube.scripts.ingest import *
from datacube.index._api import connect
from pprint import pprint
import sys

def main (argv):
    config =  "/home/cubo/configIngester/ls7_ledaps_wgs84.yaml"
    if len (argv)>1: 
        config=argv[1]
    index = ui.index_connect(application_name="agdc-ingest")
    config = load_config_from_file(index, config)
    source_type, output_type = make_output_type(index, config)
    tasks = create_task_list(index, output_type, None, source_type)
    listaURI =[]
    listaID =[]
    for task in tasks:
        listaURI.append(task['tile'].sources.values[0][0].local_uri)
        listaID.append( task['tile'].sources.values[0][0].id)
    c=connect()
    with c._db.begin() as trans:
        for idU in list(set(listaID)):
            trans._engine.execute( "DELETE FROM agdc.dataset_location WHERE agdc.dataset_location.dataset_ref ='"+idU+"'");
            trans._engine.execute( "DELETE FROM agdc.dataset WHERE agdc.dataset.id ='"+idU+"'");
        print trans.commit()
    pprint(set(listaURI));
    
if __name__ == "__main__":
    main(sys.argv)