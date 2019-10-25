# !/usr/bin/python3

"""
This module provides variables for other modules, especially:
  - a logger
  - data paths in a multiplatform manner
  - paths to specific files in a multiplatform manner
  - often used strings

NOTE:
In order to make the methods runnable, the data_path variable and
out_path must be set properly and contain all needed data
Furthermore, data_min_date and data_max_date have to be set
according to temporal data limitation
Also, given, that geographic scope is changed, lon_min, lon_max,
lat_min and lat_max have to be adjusted too

the data_path structure must have the following structure:
data_path
|_ power_load
  |_ time_series_15min_singleindex_filtered.nc
|_ ecmwf
  |_ >> here the downloaded weather data is stored
|_ shapes
  |_ NUTS_RG_60M_2016_4326_LEVL_0.shp
  |_ NUTS_RG_01M_2016_4326_LEVL_3.shp
|_ isin
  |_ >> in this folder, isin numpy 2d arrays are stored that
        are used to filter important grid points

out_path
|_ plots
  |_ >> this folder is used to store plots

Data that must be acquired before execution:
- data_path/power_load/time_series_15min_singleindex_filtered.nc
  can be downloaded from https://data.open-power-system-data.org/time_series
  NOTE, that the downloaded file is a .csv file, to convert it, use
  LoadReader's static __csv_to_nc__(file_name) method to convert it to .nc
- weather data in data_path/ecmwf
  use the ecmwf_era5_request.py script to download the data
- data_path/shapes/NUTS_RG_60M_2016_4326_LEVL_0.shp
  data_path/shapes/NUTS_RG_01M_2016_4326_LEVL_3.shp
  can be downloaded from
  https://ec.europa.eu/eurostat/de/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts
"""

__author__ = "Marcel Herm"
__credits__ = ["Marcel Herm", "Nicole Ludwig", "Marian Turowski"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Marcel Herm"

from pathlib import Path
from datetime import datetime
import os
import numpy as np
import logging

log = logging.getLogger('GISME')
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s %(message)s',
                                       datefmt='%Y-%d-%m %H:%M:%S'))
log.addHandler(handler)

# base data path, must point to path where all data is stored
data_path = os.path.join(str(Path.home()), 'Dropbox', 'data')
# figure_path = data_path + 'figures/'
load_path = os.path.join(data_path, 'power_load', 'time_series_15min_singleindex_filtered.nc')
era5_path = os.path.join(data_path, 'ecmwf')
shape_path = os.path.join(data_path, 'shapes')
isin_path = os.path.join(data_path, 'isin')
nuts0_shape = os.path.join(shape_path, 'NUTS_RG_60M_2016_4326_LEVL_0.shp')
nuts3_01res_shape = os.path.join(shape_path, 'NUTS_RG_01M_2016_4326_LEVL_3.shp')
demography_file = os.path.join(data_path, 'demo_r_pjangrp3/demo_r_pjangrp3_1_Data.csv')

# paths for output such as plots
out_path = os.path.join(str(Path.home()), 'Repos', 'GISME', 'doc')
figure_path = os.path.join(out_path, 'plots')

# often used names/labels to avoid misspelling of strings
lon_col = 'longitude'
lat_col = 'latitude'
utc_col = 'utc_timestamp'
cest_col = 'cet_cest_timestamp'
de_load = 'DE_load_actual_entsoe_transparency'
hertz_load = 'DE_50hertz_load_actual_entsoe_transparency'
amprion_load = 'DE_amprion_load_actual_entsoe_transparency'
tennet_load = 'DE_tennet_load_actual_entsoe_transparency'
transnet_load = 'DE_transnetbw_load_actual_entsoe_transparency'

# dictionary for printing human readable output for some variable names
rep_load = "actual load (MW)"  # string is repeated for each load variable, so just define once
variable_dictionary = {
    de_load: f"germany {rep_load}",
    hertz_load: f"50hertz {rep_load}",
    amprion_load: f"amprion {rep_load}",
    tennet_load: f"tennet {rep_load}",
    transnet_load: f"transnetbw {rep_load}"
}

# variables containing these words are likely to be measured in MW
# used to adjust some axes labels for plots
hasMWlbl = ['load', 'capacity', 'generation']

# geographic coordinates
lon_min = 5.5
lon_max = 15.5
lat_min = 47
lat_max = 55.5
bbox = (lon_min, lon_max, lat_min, lat_max)

numpy_funcs = {'min': np.nanmin, 'max': np.nanmax, 'mean': np.nanmean, 'median': np.nanmedian}

# minimum and maximum dates for which data is available, has to be changed different data is used
data_min_date = datetime(2015, 1, 8, 0)
data_max_date = datetime(2018, 12, 31, 23)
