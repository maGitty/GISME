#!/usr/bin/python3

from pathlib import Path

# paths
data_path = str(Path.home()) + '/Dropbox/data/'
figure_path = data_path + 'figures/'
load_path = data_path + 'power_load/time_series_15min_singleindex_filtered.csv'
utc_col = 'utc_timestamp'
cest_col = 'cet_cest_timestamp'
load_file_time_cols = [utc_col, cest_col]
era5_path = data_path + 'ecmwf/netcdf_actuals/'

# names/labels
era5_prefix = 'GridOneDayAhead_'
PKL = 'pkl'
lon = 'longitude'
lat = 'latitude'

# geographic coordinates
lon_min = 6
lon_max = 15
lat_min = 47.5
lon_max = 55
