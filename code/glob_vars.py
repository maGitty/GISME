#!/usr/bin/python3

from pathlib import Path

# paths
data_path = str(Path.home()) + '/Dropbox/data/'
#figure_path = data_path + 'figures/'
load_path = data_path + 'power_load/time_series_15min_singleindex_filtered.nc'
era5_path = data_path + 'ecmwf/'
shape_path = data_path + 'shapes/'
nuts0_shape = shape_path + 'NUTS_RG_60M_2016_4326_LEVL_0.shp'
nuts3_01res_shape = shape_path + 'NUTS_RG_01M_2016_4326_LEVL_3.shp'
nuts3_10res_shape = shape_path + 'NUTS_RG_10M_2016_4326_LEVL_3.shp'
nuts3_20res_shape = shape_path + 'NUTS_RG_20M_2016_4326_LEVL_3.shp'
nuts3_60res_shape = shape_path + 'NUTS_RG_60M_2016_4326_LEVL_3.shp'
demography_file = data_path + 'demo_r_pjangrp3/demo_r_pjangrp3_1_Data.csv'

thesis_path = str(Path.home()) + '/Repos/GISME/doc/'
figure_path = thesis_path + 'plots/'

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

