#!/usr/bin/python3

from pathlib import Path
import numpy as np

# paths
data_path = str(Path.home()) + '/Dropbox/data/'
figure_path = data_path + 'figures/'
load_path = data_path + 'power_load/time_series_15min_singleindex_filtered.nc'
era5_path = data_path + 'ecmwf/'
shape_path = data_path + 'shapes/'
nuts_shape = shape_path + 'NUTS_RG_60M_2016_4326_LEVL_0.shp'

# names/labels
era5_prefix = 'GridOneDayAhead_'
lon_col = 'longitude'
lat_col = 'latitude'
utc_col = 'utc_timestamp'
cest_col = 'cet_cest_timestamp'
de_load = 'DE_load_actual_entsoe_transparency'

hasMWlbl = ['load','capacity','generation']

# geographic coordinates
lon_min = 5.5
lon_max = 15.5
lat_min = 47
lat_max = 55.5
bbox = (lon_min,lon_max,lat_min,lat_max)

