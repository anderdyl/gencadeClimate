# GENDCADE-climate

GencadeClimate is a Python3 collection of functions for generating stochastic wave and water level scenarios for the shoreline model GENCADE.
The package creates new time series of forcing conditions by generating new possible synoptic weather chronologies. 
The workflow identifies historical synoptic weather patterns, and the meteorologic and oceanic conditions that occurred during those weather systems.
Markov chains for each pattern, as well as their likelihood of occurrence conditional dependent on large scale climate indices, are then used in monte carlo simulations.
See Anderson et al. (2019) and the references within for more details.

# Sea Level Pressure Fields
Two options for running the package: 1) direct access to ERA5 data via online connection, or 2) local download of CFSR. Specific requriement for each are outlined below.

### 1) ERA5
To use ERA5 sea level pressure fields, this package requires access to the online Thredds server hosted by Copernicus.

1. Create an account with Copernicus by signing up here.
2. Once you have an account, sign in to your Copercius account here and note the UID and API key at the bottom of the page.
3. Paste the code snippet below into your terminal, replacing 'UID' and 'API' with those from step 2:

(echo 'url: https://cds.climate.copernicus.eu/api/v2';
  echo 'key: UID:API';
  echo 'verify: 0';
   ) >> ~/.cdsapirc

The above command creates the file ~/.cdsapirc with your API key, which is necessary to use the CDS API. As a sanity check, use more ~/.cdsapirc to ensure everything appears correct.

### 2) CFSR
If CFSR is preferred, then the user must download all monthly files since 1979 found at the following links:

https://rda.ucar.edu/datasets/ds093.1/#description
https://rda.ucar.edu/datasets/ds094.1/#description

You will need to chose 'All available' under the data tab, at full resolution in lat/lon, and click the checkbox to convert the download from grib to netcdf inorder to work with the built-in functions in this library.


# Waves
Two options for running the package: 1) Wave Information Studies hindcast, or 2) direct access to ERA5 data via online connection. Specific requriement for each are outlined below.


# Climate Variables: SST and OLR
Large-scale climate is accounted for by spatial patterns of the nearby ocean's sea surface temperature (SST) pattern at an annual scale. Depending on the site of interest, the user will need to choose either 'pacificAWT' or 'atlanticAWT' from the climateIndices class



Anderson, D., A. Rueda, L. Cagigal, J. Antolinez, F. Mendez, and P. Ruggiero. (2019) Time-varying Emulator for Short and Long-Term Analysis of Coastal Flood Hazard Potential. Journal of Geophysical Research: Oceans, 124(12), 9209-9234. https://doi.org/10.1029/2019JC015312