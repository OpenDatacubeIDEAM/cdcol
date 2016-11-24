# coding=utf8
from __future__ import absolute_import
from cdcol_celery.celery import app
import time
import datacube
import numpy as np
from datacube.storage import netcdf_writer
from datacube.model import Variable, CRS
import os
import re
ALGORITHMS_FOLDER = os.path.expanduser('~')+"/algorithms"
RESULTS_FOLDER = "/Results"
nodata=-9999
@app.task
def generic_task(execID,algorithm,version, output_expression,product, min_lat, min_long, time_ranges, **kwargs):
    """
    Los primeros 8 parámetros deben ser dado por el ejecutor a partir de lo seleccionado por el usuario
        execID = id de la ejecución
        algorithm = nombre del algoritmo 
        version = versión del algoritmo a ejecutar
        output_expression = Expresión que indica cómo se va a generar el nombre del archivo de salida.
        product = producto seleccionado por el usuario (sobre el que se va a realizar la consulta)
        min_long = cordenada x de la esquina inferior izquierda del tile 
        min_lat = cordenada y de la esquina inferior izquierda del tile
        time_ranges = rangos de tiempo de las consultas (es un arreglo de tuplas, debe haber al menos una para realizar una consulta. (Obligatorio)
        kwargs = KeyWord arguments que usará el algoritmo (cuando se ejecute los verá como variables globales)
    """
    dc = datacube.Datacube(app=execID)
    i=0
    for tr in time_ranges:
        kwargs["xarr"+str(i)] = dc.load(product=product, longitude=(min_long, min_long+1.0), latitude=(min_lat, min_lat+1), time=tr)
        i+=1
    exec(open(ALGORITHMS_FOLDER+"/"+algorithm+"/"+algorithm+"_"+str(version)+".py").read(),kwargs)
    fns=[]
    folder = "{}/{}_{}/{}/".format(RESULTS_FOLDER,algorithm,str(version),execID)
    if "output" in kwargs: #output debería ser un xarray
        #Guardar a un archivo...
              
        if not os.path.exists(os.path.dirname(folder)):
            try:
                os.makedirs(os.path.dirname(folder))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        filename=folder+"{}_{}_{}.nc".format(min_lat,min_long,re.sub('[^\w_.)(-]', '', str(time_ranges)))
        nco=netcdf_writer.create_netcdf(filename)
        output=  kwargs["output"]
        coords=output.coords
        cnames=()
        for x in coords:
            netcdf_writer.create_coordinate(nco, x, coords[x].values, coords[x].units)
            cnames=cnames+(x,)
        netcdf_writer.create_grid_mapping_variable(nco, output.crs)
        for band in output.data_vars:
            output.data_vars[band].values[np.isnan(output.data_vars[band].values)]=nodata
            var= netcdf_writer.create_variable(nco, band, Variable(output.data_vars[band].dtype, nodata, cnames, None) ,set_crs=True)
            var[:] = netcdf_writer.netcdfy_data(output.data_vars[band].values)
        nco.close()
        fns.append(filename)
    if "outputtxt" in kwargs:
        if not os.path.exists(os.path.dirname(folder)):
            try:
                os.makedirs(os.path.dirname(folder))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        filename=folder+"{}_{}_{}.txt".format(min_lat,min_long,re.sub('[^\w_.)(-]', '', str(time_ranges)))
        with open(filename, "w") as text_file:
            text_file.write(kwargs["outputtxt"])
        fns.append(filename)
    return fns;
        