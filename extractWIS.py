import collections
import datetime as DT
import os
import pickle as pickle
import time
import warnings
from posixpath import join as urljoin
import socket
import netCDF4 as nc
import numpy as np
import pandas as pd
from urllib.request import urlopen
import xarray as xr
import time
from datetime import timedelta

chlDataLoc = u'https://chldata.erdc.dren.mil/thredds/dodsC/wis/'

basin = 'Atlantic'
buoy = 'ST63218'
lon = -72.25
lat = 34.75


variables = ['waveHs','waveTpPeak','waveMeanDirection']



if variables:





    t12 = time.time()

    start = DT.datetime(1980,7,1)
    end = DT.datetime(1981,3,31)


    if start.year == end.year:
        months = np.arange(start.month,end.month+1)
        counter = 0
        for hh in months:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(start.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(start.year), dataLoc)
            print('downloading {}-{}'.format(start.year,hh))
            ds = xr.open_dataset(ncfileURL)

            if counter == 0:
                df = ds[variables]
            else:
                ds2 = ds[variables]
                df = xr.concat([df,ds2],dim='time')
                counter = counter + 1

    elif start.year == (end.year-1):
        months1 = np.arange(start.month,13)
        counter = 0
        for hh in months1:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(start.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(start.year), dataLoc)
            print('downloading {}-{}'.format(start.year,hh))
            ds = xr.open_dataset(ncfileURL)

            if counter == 0:
                df = ds[variables]
            else:
                ds2 = ds[variables]
                df = xr.concat([df,ds2],dim='time')
                counter = counter + 1
        months2 = np.arange(1,end.month+1)
        for hh in months2:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(end.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(end.year), dataLoc)
            print('downloading {}-{}'.format(end.year,hh))
            ds = xr.open_dataset(ncfileURL)
            df = xr.concat([df,ds],dim='time')
            counter = counter + 1

    else:
        months1 = np.arange(start.month,13)
        counter = 0
        for hh in months1:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(start.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(start.year), dataLoc)
            print('downloading {}-{}'.format(start.year,hh))
            ds = xr.open_dataset(ncfileURL)

            if counter == 0:
                df = ds[variables]
            else:
                ds2 = ds[variables]
                df = xr.concat([df,ds2],dim='time')
                counter = counter + 1
        months2 = np.arange(1,13)
        years = np.arange(start.year+1, end.year)
        for ff in years:
            for hh in months2:
                dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(ff) + str(hh).zfill(2) + '.nc'
                ncfileURL = urljoin(chlDataLoc, basin, buoy, str(ff), dataLoc)
                print('downloading {}-{}'.format(ff, hh))
                ds = xr.open_dataset(ncfileURL)
                df = xr.concat([df, ds[variables]], dim='time')
                counter = counter + 1

        months3 = np.arange(1,end.month+1)
        for hh in months3:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(end.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(end.year), dataLoc)
            print('downloading {}-{}'.format(end.year,hh))
            ds = xr.open_dataset(ncfileURL)
            df = xr.concat([df,ds[variables]],dim='time')
            counter = counter + 1

    t22 = time.time()

    elapsed2 = t22-t12
    print('Partial WIS data files took:)')
    str(timedelta(seconds=elapsed2))



    t1 = time.time()

    start = DT.datetime(1980,7,1)
    end = DT.datetime(1981,3,31)


    if start.year == end.year:
        months = np.arange(start.month,end.month+1)
        counter = 0
        for hh in months:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(start.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(start.year), dataLoc)
            print('downloading {}-{}'.format(start.year,hh))
            ds = xr.open_dataset(ncfileURL)

            if counter == 0:
                df = ds
            else:
                df = xr.concat([df,ds],dim='time')
                counter = counter + 1

    elif start.year == (end.year-1):
        months1 = np.arange(start.month,13)
        counter = 0
        for hh in months1:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(start.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(start.year), dataLoc)
            print('downloading {}-{}'.format(start.year,hh))
            ds = xr.open_dataset(ncfileURL)

            if counter == 0:
                df = ds
            else:
                df = xr.concat([df,ds],dim='time')
                counter = counter + 1
        months2 = np.arange(1,end.month+1)
        for hh in months2:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(end.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(end.year), dataLoc)
            print('downloading {}-{}'.format(end.year,hh))
            ds = xr.open_dataset(ncfileURL)
            df = xr.concat([df,ds],dim='time')
            counter = counter + 1

    else:
        months1 = np.arange(start.month,13)
        counter = 0
        for hh in months1:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(start.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(start.year), dataLoc)
            print('downloading {}-{}'.format(start.year,hh))
            ds = xr.open_dataset(ncfileURL)

            if counter == 0:
                df = ds
            else:
                df = xr.concat([df,ds],dim='time')
                counter = counter + 1
        months2 = np.arange(1,13)
        years = np.arange(start.year+1, end.year)
        for ff in years:
            for hh in months2:
                dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(ff) + str(hh).zfill(2) + '.nc'
                ncfileURL = urljoin(chlDataLoc, basin, buoy, str(ff), dataLoc)
                print('downloading {}-{}'.format(ff, hh))
                ds = xr.open_dataset(ncfileURL)
                df = xr.concat([df, ds], dim='time')
                counter = counter + 1

        months3 = np.arange(1,end.month+1)
        for hh in months3:
            dataLoc = 'WIS-ocean_waves_' + buoy + '_' + str(end.year) + str(hh).zfill(2) + '.nc'
            ncfileURL = urljoin(chlDataLoc, basin, buoy, str(end.year), dataLoc)
            print('downloading {}-{}'.format(end.year,hh))
            ds = xr.open_dataset(ncfileURL)
            df = xr.concat([df,ds],dim='time')
            counter = counter + 1

    t2 = time.time()

    elapsed = t2-t1
    print('Full WIS data files took:')
    str(timedelta(seconds=elapsed))


