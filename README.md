# GENDCADE-climate

GencadeClimate is a Python3 collection of functions for generating stochastic wave and water level scenarios for the shoreline model GENCADE.
The package creates new time series of forcing conditions by generating new possible synoptic weather chronologies. 
The workflow identifies historical synoptic weather patterns, and the meteorologic and oceanic conditions that occurred during those weather systems.
Markov chains for each pattern, as well as their likelihood of occurrence conditional dependent on large scale climate indices, are then used in monte carlo simulations.
See Anderson et al. (2019) and the references within for more details.


To use ERA5 sea level pressure fields, this package requires access to the online Thredds server hosted by Copernicus.

1. Create an account with Copernicus by signing up here.
2. Once you have an account, sign in to your Copercius account here and note the UID and API key at the bottom of the page.
3. Paste the code snippet below into your terminal, replacing 'UID' and 'API' with those from step 2:

(echo 'url: https://cds.climate.copernicus.eu/api/v2';
  echo 'key: UID:API';
  echo 'verify: 0';
   ) >> ~/.cdsapirc

The above command creates the file ~/.cdsapirc with your API key, which is necessary to use the CDS API. As a sanity check, use more ~/.cdsapirc to ensure everything appears correct.



Anderson, D., A. Rueda, L. Cagigal, J. Antolinez, F. Mendez, and P. Ruggiero. (2019) Time-varying Emulator for Short and Long-Term Analysis of Coastal Flood Hazard Potential. Journal of Geophysical Research: Oceans, 124(12), 9209-9234. https://doi.org/10.1029/2019JC015312