# GISME
GIS meets Energy

This package provides variables for other modules, especially:
  - a logger
  - data paths in a multiplatform manner
  - paths to specific files in a multiplatform manner
  - often used strings

NOTE:
In order to make the methods runnable, the data_path variable and out_path must be set properly and contain all needed data. Furthermore, data_min_date and data_max_date have to be set according to temporal data limitation. Also, given, that geographic scope is changed, lon_min, lon_max, lat_min and lat_max have to be adjusted too the data_path structure must have the following structure:  

data_path  
|_ power_load  
--|_ time_series_15min_singleindex_filtered.nc  
|_ ecmwf  
--|_ >> here the downloaded weather data is stored  
|_ shapes  
--|_ NUTS_RG_60M_2016_4326_LEVL_0.shp  
--|_ NUTS_RG_01M_2016_4326_LEVL_3.shp  
|_ isin  
--|_ >> in this folder, isin numpy 2d arrays are stored that are used to filter important grid points out_path  
|_ plots  
--|_ >> this folder is used to store plots  

Data that must be acquired before execution:
- data_path/power_load/time_series_15min_singleindex_filtered.nc
  can be downloaded from https://data.open-power-system-data.org/time_series
  NOTE, that the downloaded file is a .csv file, to convert it, use
  LoadReader's static \__csv_to_nc__(file_name) method to convert it to .nc
- weather data in data_path/ecmwf
  use the ecmwf_era5_request.py script to download the data
- data_path/shapes/NUTS_RG_60M_2016_4326_LEVL_0.shp
  data_path/shapes/NUTS_RG_01M_2016_4326_LEVL_3.shp
  can be downloaded from
  https://ec.europa.eu/eurostat/de/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts
