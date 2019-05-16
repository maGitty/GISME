#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:05:46 2019

@author: Nicole Ludwig

File to extract the ECMWF ensemble data needed for forecasting the national grid demand in the UK
Extract the 50 perturbed forecasts as well as the control forecast for a seven day ahead horizon (12 am/pm)
and a 6 day ahead horizon (6am/pm)

Needs full MARS access!

Output are individual csv and nc files for 7 forecast origins per month
"""

import os
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from ecmwfapi import ECMWFService


def days_of_month(y, m):
    d0 = datetime(y, m, 1)
    d1 = d0 + relativedelta(months=1)
    out = list()
    while d0 < d1:
        out.append(d0.strftime('%Y-%m-%d'))
        d0 += timedelta(days=1)
    return out



server = ECMWFService("mars")

target_dir = str(Path.home()) + "/Forecast/ecmwf/"
target_out = "GridEnsembles"
target_dwl = target_dir + "ensemble_netcdf/"
if not os.path.exists(target_dir + target_out):
    os.makedirs(target_dir + target_out)

if not os.path.exists(target_dwl):
    os.makedirs(target_dwl)
# path_data = '/home/ws/ob4015/Forecast/ecmwf/'
# path_output = '/home/ws/ob4015/Forecast/ecmwf/GridEnsembles/'

years = range(2018, 2019)
months = range(1, 2)

# start by extracting the actual data for the whole time period, data sets split into years (and months?)

for y in years:
    for m in months:
        for i in range(1, 8):
            d = days_of_month(y, m)[(i-1)::7]

            server.execute(
                {
                    'class': "od",  # operational data
                    'date': d,
                    'expver': 1,
                    'levtype': "sfc",  # surface level
                    'number': '1/to/50',  # all ensemble members
                    'param': ["20.3", "164.128", "165.128", "166.128", "167.128", "186.128", "228.128"],
                    'step': [24, 48, 72, 96, 120, 144],  # 168 only available for 12am/pm
                    'stream': "enfo",  # ensemble forecast
                    'time': ["00:00:00", '06:00:00', '12:00:00', '18:00:00'],
                    'type': "pf",  # perturbed forecast
                    'area': '58.6883/-8.4814/49.8096/1.8237',  # coordinates UK
                    'grid': '0.25/0.25',  # grid size in degree
                    'format': 'netcdf'  # get a net cdf file
                },
                target_dwl + target_out + '_{date}.nc'.format(date=d[0])
            )

            server.execute(
                {
                    'class': "od",  # operational data
                    'date': d,
                    'expver': 1,
                    'levtype': "sfc",  # surface level
                    'param': ["20.3", "164.128", "165.128", "166.128", "167.128", "186.128", "228.128"],
                    'step': [24, 48, 72, 96, 120, 144],  # 168 only available for 12am/pm
                    'stream': "enfo",  # ensemble forecast
                    'time': ["00:00:00", '06:00:00', '12:00:00', '18:00:00'],
                    'type': "cf",  # control forecast
                    'area': '58.6883/-8.4814/49.8096/1.8237',  # coordinates UK
                    'grid': '0.25/0.25',  # grid size in degree
                    'format': 'netcdf'  # get a net cdf file
                },
                target_dwl + target_out + '_cf_{date}.nc'.format(date=d[0])
            )


# transform netcdf files to csv files

files = []

for f in os.listdir(target_dwl):
    if f.endswith(".nc"):
        files.append(f)

for f in files:
    os.chdir(target_dir + target_out)

    if len(f) == 29:
        print(f)

        ds = xr.open_dataset(f)
        df = ds.to_dataframe()
        df.to_csv('ensembles_' + f[16:26] + '.csv')

    elif len(f) == 32:
        print(f)

        ds = xr.open_dataset(f)
        df = ds.to_dataframe()
        df.to_csv('control_' + f[19:29] + '.csv')
