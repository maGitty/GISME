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
import glob
import pandas as pd
import re
import cdsapi
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


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



target_dir = str(Path.home()) + "/Forecast/ecmwf/"
target_out = "GridActuals"
target_dwl = target_dir + 'netcdf_files/'

if not os.path.exists(target_dir + target_out):
    os.makedirs(target_dir + target_out)

if not os.path.exists(target_dwl):
    os.makedirs(target_dwl)

years = range(2014, 2018) # for later: everything between 24.04.2011 and 31.12.2011 is missing for now, do something like range(2006, 2019)?
months = range(1, 13) # normally range(1, 13)

# functions needed to get the individual days of each month and to properly
# arrange the files according to their alphanumerical name

client = cdsapi.Client()


# start by extracting the actual data for the whole time period, retrieve daily data set
# for y in range(2006, 2018):  # for later: everything between 24.04.2011 and 31.12.2011 is missing for now
for y in years:
    for m in months:
        for day in days_of_month(y, m):
            if not os.path.exists(target_dwl + 'GridOneDayAhead_{date}.nc'.format(date=day)):
                client.retrieve("reanalysis-era5-single-levels",
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

# transform netcdf files to csv files
files = []

# append the individual daily nc files to one csv file per year
for f in os.listdir(target_dwl):
    if f.endswith(".nc"):
        files.append(f)

# sort list of file names
files.sort(key=tokenize)

for y in years:
    results = pd.DataFrame()
    y = str(y)
    os.chdir(target_dwl)
    for f in files:
        if f in glob.glob(target_dwl + 'GridOneDayAhead_' + y + '*.nc'):
            print(f)
            ds = xarray.open_dataset(f)
            df = ds.to_dataframe()
            results = results.append(df, sort=False)

    results.to_csv(target_dir + 'GridActuals_' + y + '.csv')
