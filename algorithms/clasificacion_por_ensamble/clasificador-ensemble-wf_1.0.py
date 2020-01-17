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
bands_data = []

print('medians1.datavars',medians1.data_vars.keys())

#rows, cols = xarr0[product['bands'][0]].shape


bands_data=[]

bands2=list(medians1.data_vars.keys())
print('bands2',bands2)


for band in bands2: 
	if band != 'pixel_qa':
   	 bands_data.append(medians1[band])
bands_data = np.dstack(bands_data)

print('bands_data_test2',bands_data)


rows, cols, n_bands = bands_data.shape

print('rows',rows)
print('cols',cols)

print('n_bands',n_bands)



n_samples = rows*cols
flat_pixels = bands_data.reshape((n_samples, n_bands))
#mascara valores nan por valor no data
mask_nan=np.isnan(flat_pixels)
flat_pixels[mask_nan]=-9999
#result = bagging_clf.predict(flat_pixels)
#classification = result.reshape((rows, cols))
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
    bagging_clf = joblib.load(model)
    print(bagging_clf)

#print(bagging_clf)
n_samples = rows*cols
flat_pixels = bands_data.reshape((n_samples, n_bands))
#mascara valores nan por valor no data
mask_nan=np.isnan(flat_pixels)
flat_pixels[mask_nan]=-9999

print('flat_pixels')
print(type(flat_pixels))
print(flat_pixels.min())
print(flat_pixels.max())
print(flat_pixels)


from numpy import inf
                        
flat_pixels[flat_pixels<0]=0
flat_pixels[flat_pixels==-inf]= 0
flat_pixels[flat_pixels>= 1E308]=0
                                    

print('flat_pixels 3')
print(type(flat_pixels))
print(flat_pixels.min())
print(flat_pixels.max())
print(flat_pixels)                          
#np.isfinite(flat_pixels)

_msk=np.sum(np.isfinite(flat_pixels),1)>1
#flat_pixels= flat_pixels[_msk,:]
flat_pixels=flat_pixels[_msk]


print(flat_pixels.min())
print(flat_pixels.max())                      
#result = bagging_clf.predict(flat_pixels)
#classification = result.reshape((rows, cols))

print("clasificacion final")
result = bagging_clf.predict(flat_pixels)
result = result.reshape((rows, cols))
print("fin funcion de clasificacion")
print(result)

print('tipo result')
print(type(result))

#result = bagging_clf.predict(nmed.T)
#result = result.reshape(sp)

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
