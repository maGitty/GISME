#!/usr/bin/python3

from glob_vars import data_path

import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, dates
from datetime import datetime, timezone, timedelta


class WeatherRelator:
    """
    
    """

    def nmin_val_days(var, num=4):
        """Calculate the num days with lowest temperature for var
        Parameters
        ----------
        var : str
              name of the variable
        num : numeric, default 4
              number of values
        """
        pass
    
    def nmax_var_days(var, num=4):
        """Calculate the num days with highest variance for var
        Parameters
        ----------
        var : str
              name of the variable
        num : numeric, default 4
              number of values
        """
        pass        

def plot_temp_maxVSload_attime(hour):
    """
    @hour: time string with format HH:MM:SS
    get max temp for each day at same time and plot vs load
    """
    #years = range(2014, 2019) 
    #lons = np.arange(6, 15.25, 0.25)
    #lats = np.arange(47.5, 55.25, .25)
    #coords = [(x,y) for x in lons for y in lats]
    drang = pd.date_range(start=f'1/1/2017 {hour}', end=f'31/12/2017 {hour}', freq='1D')
    
    weather_df = pd.read_csv(data_path + 'ecmwf/GridActuals_2017.csv', low_memory=False)
    weather_df['t2m'] = weather_df['t2m'].apply(lambda x: x - 273.15)
    weather_df = weather_df[['time', 'latitude', 'longitude', 't2m']]
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    
    times = []
    temp_values = []
    for t in drang:
        point_df = weather_df[weather_df['time'] == t]
        tval = point_df['t2m'].max()
        temp_values.append(tval)
        tstamp = point_df.loc[point_df['t2m'] == tval, 'time'].iat[0]
        #tstamp = tstamp.tz_convert('Europe/Berlin')
        times.append(tstamp)
    
    print('done')
    power_df = pd.read_csv(data_path + "power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
    power_df = power_df[power_df[de_load].notnull()]
    power_df[time_col] = pd.to_datetime(power_df[time_col], format='%Y-%m-%dT%H:%M:%SZ')
    
    load_values = []
    for t in times:
        vload = power_df.loc[power_df[time_col] == t, de_load].iloc[0]
        load_values.append(vload)
    
    plt.plot(temp_values, load_values, 'o')
    plt.xlabel('DE max temp [°C]')
    plt.ylabel('DE load [MW]')
    plt.show()
        
def check():
    weather_df = pd.read_csv(data_path + 'ecmwf/GridActuals_2017.csv', low_memory=False)
    weather_df = weather_df[weather_df['time'] == '2017-06-01 12:00:00']
    #print(weather_df['longitude'].max())
    #print(weather_df['longitude'].min())
    #print(weather_df['latitude'].max())
    #print(weather_df['latitude'].min())
    
    lons = np.arange(6, 15.25, 0.25)
    lats = np.arange(47.5, 55.25, .25)
    coords = [(x,y) for x in lons for y in lats]
    print(coords)


def check_temp():
    weather_df = pd.read_csv(data_path + 'ecmwf/GridActuals_2017.csv', low_memory=False)
    print(len(weather_df))
    

#drang = pd.date_range(start='1/1/2015', end='30/04/2019', freq='15T', tz='Europe/Berlin')
#print(drang[1])
#print(drang[-2])

time_col = 'utc_timestamp'
de_load = 'DE_load_actual_entsoe_transparency'
hertz_load = 'DE_50hertz_load_actual_entsoe_transparency'
amprion_load = 'DE_amprion_load_actual_entsoe_transparency'
tennet_load = 'DE_tennet_load_actual_entsoe_transparency'
transnetbw_load = 'DE_transnetbw_load_actual_entsoe_transparency'

plot_temp_maxVSload_attime('12:00:00')
plot_temp_maxVSload_attime('18:00:00')
