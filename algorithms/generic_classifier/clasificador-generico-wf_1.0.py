# In[7]:
import xarray as xr
import numpy as np
from sklearn.externals import joblib
import warnings

# In[21]:

# Preprocesar:
nmed=None
nan_mask=None
xarrs=list(xarrs.values())
print(type(xarrs))
medians1 = xarrs[0]
print(type(medians1))
print("medianas")
for band in medians1.data_vars.keys():
    if band == "crs":
        continue
    b=np.ravel(medians1.data_vars[band].values)
    if nan_mask is None:
        nan_mask=np.isnan(b)
    else:
        nan_mask=np.logical_or(nan_mask, np.isnan(medians1.data_vars[band].values.ravel()))
    b[np.isnan(b)]=np.nanmedian(b)
    if nmed is None:
        sp=medians1.data_vars[band].values.shape
        nmed=b
    else:
        nmed=np.vstack((nmed,b))

# In[12]:

import os

print("modelo")

model = None
for file in other_files:
    if file.endswith(".pkl"):
        model = file
        break
if model is None:
    raise "Debería haber un modelo en la carpeta " + modelos

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    classifier = joblib.load(model)

print(classifier)
print('nmedT',nmed.T)
result = classifier.predict(nmed.T)
print('result1',result)

result = result.reshape(sp)
print('result2',result)


# In[ ]:


# In[24]:

coordenadas = []
dimensiones = []
xcords = {}
for coordenada in xarrs[0].coords:
    if (coordenada != 'time'):
        coordenadas.append((coordenada, xarrs[0].coords[coordenada]))
        dimensiones.append(coordenada)
        xcords[coordenada] = xarrs[0].coords[coordenada]

valores = {"classified": xr.DataArray(result, dims=dimensiones, coords=coordenadas)}
#array = xr.DataArray(result, dims=dimensiones, coords=coordenadas)
#array.astype('float32')
#valores = {"classified": array}

output = xr.Dataset(valores, attrs={'crs': xarrs[0].crs})
for coordenada in output.coords:
    output.coords[coordenada].attrs["units"] = xarrs[0].coords[coordenada].units

classified = output.classified
classified.values = classified.values.astype('float32')
