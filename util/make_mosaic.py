# coding: utf-8
#requires xarray >1.9 y pandas > 0.17
# In[5]:

import xarray as xr

import glob, os,sys
folder=sys.argv[1]
postfix=sys.argv[2]
os.chdir(folder)
output=None
for file in glob.glob("*_{}.nc".format(postfix)):
    if(output is None):
        output=xr.open_dataset(file)
    else:
        output=output.combine_first(xr.open_dataset(file))
print output


# In[6]:

output.to_netcdf('mosaico_xarr_{}.nc'.format(postfix))


# In[ ]:



