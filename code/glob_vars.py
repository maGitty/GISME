#!/usr/bin/python3

from pathlib import Path

# paths
data_path = str(Path.home()) + '/Dropbox/data/'
figure_path = data_path + 'figures/'
load_path = data_path + 'power_load/time_series_15min_singleindex_filtered.csv'
era5_path = data_path + 'ecmwf/'
shape_path = data_path + 'shapes/'

# names/labels
era5_prefix = 'GridOneDayAhead_'
PKL = 'pkl'
lon = 'longitude'
lat = 'latitude'
utc_col = 'utc_timestamp'
cest_col = 'cet_cest_timestamp'

# geographic coordinates
lon_min = 5.5
lon_max = 15.5
lat_min = 47
lat_max = 55.5
