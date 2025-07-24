#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
Retrieve a subset of AZMP data.
'''

# Module
import numpy as np
import pandas as pd
import os
import xarray as xr

## MAIN
# Buoy information
buoy    = 'iml-4'
year    = '2023'
cbd     = '2023-08-22T00:00:00'
ced     = '2023-08-31T23:59:59'
lvl     = '1'
lvl_id  = 'wavespectra'

# Make input directory and file, and output file
lvl_dir  = '../ProcessLVL/%s/lvl%s/'%(buoy,lvl)
lvl_file = '%s_lvl%s_%s_%s.nc'%(buoy,lvl,lvl_id,year)
out_dir  = 'lvl%s_subsetted/'%lvl
out_file = '%s_lvl%s_%s_%s_%s_subsetted.nc'\
           %(buoy,lvl,lvl_id,cbd.split('T')[0],ced.split('T')[0])

# Load data for given buoy and level
DS = xr.open_dataset(lvl_dir+lvl_file,engine='netcdf4')

# Retrieve cropped time indices
time = np.array([pd.Timestamp(t).timestamp() for t in DS.Time.values])
i0   = abs(time-pd.Timestamp(cbd).timestamp()).argmin()
i1   = abs(time-pd.Timestamp(ced).timestamp()).argmin()

# Output cropped dataset
os.system("bash -c '%s'"%('mkdir -p %s 2>/dev/null'%out_dir))
DSo = DS.isel(Time=slice(i0,i1))
DSo.to_netcdf(out_dir+out_file,engine='netcdf4')

# END
