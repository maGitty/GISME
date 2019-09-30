#!/usr/bin/python3

"""
This module provides variables for other modules, especially:
  - a logger
  - data paths in a multiplatform manner
  - paths to specific files in a multiplatform manner
  - often used strings
"""

__author__ = "Marcel Herm"
__credits__ = ["Marcel Herm","Nicole Ludwig","Marian Turowski"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Marcel Herm"
__status__ = "Production"

from pathlib import Path
import os
import logging

log = logging.getLogger('GISME')
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s %(message)s',
                                       datefmt='%Y-%d-%m %H:%M:%S'))
log.addHandler(handler)

# paths
data_path = os.path.join(str(Path.home()),'Dropbox','data')
#figure_path = data_path + 'figures/'
load_path = os.path.join(data_path,'power_load','time_series_15min_singleindex_filtered.nc')
era5_path = os.path.join(data_path,'ecmwf')
shape_path = os.path.join(data_path,'shapes')
isin_path = os.path.join(data_path,'isin')
nuts0_shape = os.path.join(shape_path,'NUTS_RG_60M_2016_4326_LEVL_0.shp')
nuts0_10res_shape = os.path.join(shape_path,'NUTS_RG_10M_2016_4326_LEVL_0.shp')
nuts3_01res_shape = os.path.join(shape_path,'NUTS_RG_01M_2016_4326_LEVL_3.shp')
nuts3_10res_shape = os.path.join(shape_path,'NUTS_RG_10M_2016_4326_LEVL_3.shp')
nuts3_20res_shape = os.path.join(shape_path,'NUTS_RG_20M_2016_4326_LEVL_3.shp')
nuts3_60res_shape = os.path.join(shape_path,'NUTS_RG_60M_2016_4326_LEVL_3.shp')
demography_file = os.path.join(data_path,'demo_r_pjangrp3/demo_r_pjangrp3_1_Data.csv')

thesis_path = os.path.join(str(Path.home()),'Repos','GISME','doc')
figure_path = os.path.join(thesis_path,'plots')

# names/labels
era5_prefix = 'GridOneDayAhead_'
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
rep_load = "actual load (MW)" # string is repeated for each load variable, so just define once
variable_dictionary = {
    de_load : f"germany {rep_load}",
    hertz_load : f"50hertz {rep_load}",
    amprion_load : f"amprion {rep_load}",
    tennet_load : f"tennet {rep_load}",
    transnet_load : f"transnetbw {rep_load}"
}

# variables containing these words are likely to be measured in MW
hasMWlbl = ['load','capacity','generation']

# geographic coordinates
lon_min = 5.5
lon_max = 15.5
lat_min = 47
lat_max = 55.5
bbox = (lon_min,lon_max,lat_min,lat_max)

