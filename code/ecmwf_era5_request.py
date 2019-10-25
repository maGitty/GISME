#!/usr/bin/python3

"""
This module provides functionality to download the needed weather data
"""

from gisme import (era5_path, lon_min, lon_max, lat_min, lat_max)

import os
import cdsapi
import urllib3
import threading

# to suppress InsecureRequestWarning thrown when using mobile network or some sorts of networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def retrieve_year():
    while True:
        # if list with pairs of year and list of months is empty, exit loop
        if not yms:
            print('list is empty')
            break
        year, month = yms.pop()
        # set up paths
        ncfile = f'ERA5_RSL_{year}{month}.nc'
        file_path = era5_path+ncfile
        # check if file already exists, if so, continue with next file
        if os.path.exists(file_path):
            print(f'{ncfile} already exists')
            continue
        try:
            client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                        'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 'low_cloud_cover',
                        'soil_temperature_level_1', 'surface_latent_heat_flux', 'surface_net_thermal_radiation',
                        'surface_sensible_heat_flux', 'total_cloud_cover', 'total_column_rain_water',
                        'total_sky_direct_solar_radiation_at_surface'
                    ],
                    'year': [
                        str(year)
                    ],
                    'month': [
                        month
                    ],
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31'
                    ],
                    'area': [
                        lat_max, lon_min, lat_min, lon_max  # N, W, S, E
                    ],
                    'grid': [
                        0.25, 0.25
                    ],
                    'time': [
                        '00:00', '02:00', '04:00',
                        '06:00', '08:00', '10:00',
                        '12:00', '14:00', '16:00',
                        '18:00', '20:00', '22:00'
                    ],     
                    'format': 'netcdf'
                },
                f'{era5_path}{ncfile}')
        except:
            print(f'download not available for {year} in month {month}')


def download_ecmwf_data():

    # create folder to store data if it does not exist yet
    if not os.path.exists(era5_path):
        os.makedirs(era5_path)

    ###################
    # retrieving data #
    ###################
    # may be used too, but actually 10 is a good number as copernicus allows up to 10 threads
    # num_threads = os.cpu_count()
    threads = []

    for i in range(10):
        t = threading.Thread(target=retrieve_year)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


######################################
# uncomment the following section to #
# download weather data from ecmwf   #
######################################

client = cdsapi.Client()
"""
yms is a list of tuples, each tuple containing a year and a month;
the data is split into single months, because the size of files
that are downloaded is limited by copernicus
"""
years = range(2015, 2019)
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
yms = []
for year in years:
    for month in months:
        yms.append((str(year), month))

download_ecmwf_data()
