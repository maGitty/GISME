#!/usr/bin/python3
from glob_vars import era5_path, lon_min, lon_max, lat_min, lat_max

import os
import cdsapi
import urllib3
import threading

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # to suppress InsecureRequestWarning thrown when using mobile network or similar
if not os.path.exists(era5_path):
    os.makedirs(era5_path)

def retrieve_year():
    while True:
        # if list with pairs of year and list of months is empty, exit loop
        if not yms:
            print('list is empty')
            break
        year,month = yms.pop()
        # set up paths
        month_ind = 1 if month[0] == '01' else 2
        ncfile = f'ERA5_RSL_{year}H{month_ind}.nc'
        file_path = era5_path+ncfile
        # check if file already exists, if so, continue with next file
        if os.path.exists(file_path):
            print(f'{ncfile} already exists')
            continue
        try:
            client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type':'reanalysis',
                    'variable':[
                        '10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature',
                        'evaporation','instantaneous_moisture_flux','k_index',
                        'low_cloud_cover','skin_temperature','surface_net_thermal_radiation',
                        'surface_pressure','total_cloud_cover','total_column_water_vapour',
                        'total_precipitation'
                    ],
                    'year':[
                        str(year)
                    ],
                    'month':month,
                    'day':[
                        '01','02','03',
                        '04','05','06',
                        '07','08','09',
                        '10','11','12',
                        '13','14','15',
                        '16','17','18',
                        '19','20','21',
                        '22','23','24',
                        '25','26','27',
                        '28','29','30',
                        '31'
                    ],
                    'area':[
                        55.5,5.5,47,
                        15.5
                    ],
                    'grid':[
                        0.25,0.25
                    ],
                    'time':[
                        '00:00','06:00','12:00',
                        '18:00'
                    ],     
                    'format':'netcdf'
                },
                file_path)
        except:
            print(f'download not available for {year} in {month_ind} half')
      
            
client = cdsapi.Client()

years = range(2015,2020)
months = [['01','02','03','04','05','06'],['07','08','09','10','11','12']]
# yms consists of a year and a list of months of one half of a year
# because a the cdsapi refused downloading due to a size limit
yms = []
for year in years:
    for month in months:
        yms.append((str(year),month))
yms.pop() # pop first, which is second half of 2019 for which no data exists yet      

###################
# retrieving data #
###################
num_threads = os.cpu_count()
threads = []

for i in range(10):
    t = threading.Thread(target=retrieve_year)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
###################
