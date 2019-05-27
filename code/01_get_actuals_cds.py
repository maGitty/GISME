#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:05:46 2019

@author: Nicole Ludwig

File to extract the ERA- 5 Copernicus (ECMWF) data needed for forecasting the national grid demand
In this part extract the 'actual' data on the surface level

Needs an account at the CDS climate storage (https://cds.climate.copernicus.eu/api-how-to)

"""

import os
import xarray
import pandas as pd
import re
import time
import cdsapi
import threading
import urllib3
from glob import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from glob_vars import data_path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # to suppress InsecureRequestWarning thrown when using mobile network or similar

def days_of_month(y, m):
    d0 = datetime(y, m, 1)
    d1 = d0 + relativedelta(months=1)
    out = list()
    while d0 < d1:
        out.append(d0.strftime('%Y-%m-%d'))
        d0 += timedelta(days=1)
    return out


def tokenize(filename):
    digits = re.compile(r'(\d+)')
    return tuple(int(token) if match
                 else token for token, match in ((fragment, digits.search(fragment))
                            for fragment in digits.split(filename)))


# start by extracting the actual data for the whole time period, retrieve daily data set
def retrieve_next():
    while True:
        if not dates:
            print("date list is empty, exiting...")
            break
        day = dates.pop()
        if not os.path.exists(target_dwl + 'GridOneDayAhead_{date}.nc'.format(date=day)):
            print("sending request for {date}".format(date=day))
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    'product_type': 'reanalysis',  # reanalysis are sort of the actual values
                    'date': day,  # range as specified above
                    # the parameters can be specified using names or numbers check the webpage for the names
                    'param': ["20.3", "164.128", "165.128", "166.128", "167.128", "186.128", "228.128"],
                    'time': ['00:00:00', '06:00:00', '12:00:00', '18:00:00'],  # data origin (more possible)
                    'area': [55.099161, 5.8663153, 47.2701114, 15.0419319],  # coordinates UK [n, w, s, e] or upper left and lower right corner
                    'grid': [0.25, 0.25],  # grid size in degree (ca. 25km grid)
                    'format': 'netcdf',  # get a netcdf file (easier to handle than grib)
                },
                target_dwl + 'GridOneDayAhead_{date}.nc'.format(date=day)
                )
        else:
            pass # print("File for {date} already exists".format(date=day))


def concat_year():
    while True:
        if not year_list:
            print("year list is empty, exiting...")
            break
        y = str(year_list.pop())
        glob_list = glob('GridOneDayAhead_' + y + '*.nc')
        results = pd.DataFrame()
        for f in files:
            if f in glob_list:
                ds = xarray.open_dataset(f.split('/')[-1])
                df = ds.to_dataframe()
                results = results.append(df, sort=False)
        print("concatenated file for {year}".format(year=y))
        results.to_csv(target_dir + 'GridActuals_' + y + '.csv')

target_dir = data_path + 'ecmwf/'
target_dwl = target_dir + 'netcdf_actuals/'

if not os.path.exists(target_dwl):
    os.makedirs(target_dwl)

years = range(2006, 2019) # for later: everything between 24.04.2011 and 31.12.2011 is missing for now, do something like range(2006, 2019)?
months = range(1, 13) # normally range(1, 13)
year_list = list(years)

dates = []

for year in years:
    for month in months:
        for day in days_of_month(year, month):
            dates.append(day)

# functions needed to get the individual days of each month and to properly
# arrange the files according to their alphanumerical name

client = cdsapi.Client()

threads = []

###################
# retrieving data #
###################
for i in range(10):
    t = threading.Thread(target=retrieve_next)
    t.start()
    threads.append(t)
    # time.sleep(1) # sleep needed to put some space between requests?!

for t in threads:
    t.join()
###################

# transform netcdf files to csv files
files = []

##########################
# concat nc files to csv #
##########################
for f in os.listdir(target_dwl):
    if f.endswith(".nc"):
        files.append(f)

# sort list of file name s
files.sort(key=tokenize)

os.chdir(target_dwl)
num_cores = os.cpu_count()

print("starting {cores} threads for {years} files".format(cores=num_cores, years=len(year_list)))

threads = []
for i in range(num_cores):
    t = threading.Thread(target=concat_year)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
