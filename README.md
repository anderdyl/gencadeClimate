# GENDCADE-climate

GencadeClimate is a Python3 collection of functions for generating stochastic wave and water level scenarios for the shoreline model GENCADE.

1. Create an account with Copernicus by signing up here.
2. Once you have an account, sign in to your Copercius account here and note the UID and API key at the bottom of the page.
3. Paste the code snippet below into your terminal, replacing <UID> and <API key> with those from step 2:
   (
  echo 'url: https://cds.climate.copernicus.eu/api/v2';
  echo 'key: 139138:e2bc4589-a9ff-4b26-a408-fcf172ff0f6b';
  echo 'verify: 0';
   ) >> ~/.cdsapirc
The above command creates the file ~/.cdsapirc with your API key, which is necessary to use the CDS API. As a sanity check, use more ~/.cdsapirc to ensure everything appears correct.

