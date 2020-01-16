import xarray as xr
import numpy as np
import collections

print("Ejecutando calculo de series de tiempo en WOFS")

def perform_timeseries_analysis(dataset_in, no_data=-9999):
    """
    Description:

    -----
    Input:
      dataset_in (xarray.DataSet) - dataset with one variable to perform timeseries on
    Output:
      dataset_out (xarray.DataSet) - dataset containing
        variables: normalized_data, total_data, total_clean
    """

    data_vars = dataset_in.data_vars
    key = "wofs"

    data = data_vars[key]
    data.values[np.isnan(data.values)] = no_data
    # shape = data.shape[1:]

    data_dup = data.copy(deep=True)
    data_dup.values = data_dup.values.astype('float')
    data_dup.values[data.values == no_data] = 0

    processed_data_sum = data_dup.sum('time')
    del data_dup
    # Masking no data values then converting boolean to int for easy summation
    clean_data_raw = np.reshape(np.in1d(data.values.reshape(-1), [no_data], invert=True),
                                data.values.shape).astype(int)
    # Create xarray of data
    time = data.time
    print(data.coords)
    print(data.dims)
    if hasattr(data, "latitude"):
        latitude = data.latitude
        longitude = data.longitude
        clean_data = xr.DataArray(clean_data_raw,
                                  coords=data.coords,
                                  dims=data.dims)
    else:
        y = data.y
        x = data.x
        clean_data = xr.DataArray(clean_data_raw,
                                  coords=[time, y, x],
                                  dims=['time', 'y', 'x'])

    clean_data_sum = clean_data.sum('time')

    processed_data_normalized = processed_data_sum/clean_data_sum
    if hasattr(data, "latitude"):
        dataset_out = xr.Dataset(collections.OrderedDict(
            [('normalized_data', (['latitude', 'longitude'], processed_data_normalized.astype(np.float32))),
             ('total_data', (['latitude', 'longitude'], processed_data_sum.astype(np.int16))),
             ('total_clean', (['latitude', 'longitude'], clean_data_sum.astype(np.int16)))]),
                                 coords={'latitude': latitude,
                                         'longitude': longitude})
    else:
        dataset_out = xr.Dataset(
            collections.OrderedDict([('normalized_data', (['y', 'x'], processed_data_normalized.astype(np.float32))),
                                     ('total_data', (['y', 'x'], processed_data_sum.astype(np.int16))),
                                     ('total_clean', (['y', 'x'], clean_data_sum.astype(np.int16)))]),
            coords={'y': y,
                    'x': x})
    return dataset_out

crs_org = xarr0.crs
time_series = perform_timeseries_analysis(xarr0)
outputs={}
outputs["time_series"]=time_series
outputs["time_series"].attrs["crs"]=crs_org
