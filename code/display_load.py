from glob_vars import data_path

from matplotlib import pyplot as plt, dates
from datetime import datetime, timezone, timedelta
import csv
import pandas as pd
import numpy as np
import pytz


def cleanfor_DE_load():
    power_df = pd.read_csv(data_path + "power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
    power_df = power_df[power_df[de_load].notnull()]
    power_df[time_col] = pd.to_datetime(power_df[time_col], format='%Y-%m-%dT%H:%M:%S%z')
    start_time = datetime(2015, 1, 1, tzinfo=pytz.timezone('Europe/Berlin'))
    power_df = power_df[power_df[time_col] >= start_time]
    power_df[[time_col, de_load]].to_csv('/home/marcel/test.csv', ',')


def plot_line():
    # utc_timestamp, cet_cest_timestamp
    power_df = pd.read_csv(data_path + "power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
    power_df = power_df[power_df[de_load].notnull()]
    power_df[time_col] = pd.to_datetime(power_df[time_col], format='%Y-%m-%dT%H:%M:%S%z')
    start_time = datetime(2015, 1, 1, tzinfo=pytz.timezone('Europe/Berlin'))
    power_df = power_df[power_df[time_col] >= start_time]
    dates = power_df[time_col]
    #print(dates.where(dates.notna()).head())
    data = power_df.where(power_df[de_load].notna())[de_load]
    plt.plot(dates, data, '-', linewidth=.5)
    plt.xlabel(time_col)
    plt.ylabel('load[MW]')
    plt.show()


def plot_all():
    # utc_timestamp, cet_cest_timestamp
    power_df = pd.read_csv(data_path + "power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
    power_df = power_df[power_df[de_load].notnull()]
    power_df[time_col] = pd.to_datetime(power_df[time_col], format='%Y-%m-%dT%H:%M:%S%z')
    start_time = datetime(2015, 1, 1, tzinfo=pytz.timezone('Europe/Berlin'))
    power_df = power_df[power_df[time_col] >= start_time]    
    dates = power_df[time_col]
    
    de_load_data = power_df.where(power_df[de_load].notna())[de_load]
    hertz_load_data = power_df.where(power_df[hertz_load].notna())[hertz_load]
    amprion_load_data = power_df.where(power_df[amprion_load].notna())[amprion_load]
    tennet_load_data = power_df.where(power_df[tennet_load].notna())[tennet_load]
    transnetbw_load_data = power_df.where(power_df[transnetbw_load].notna())[transnetbw_load]
    
    fig = plt.figure(4)
    ax = fig.add_subplot(211)    
    
    linewidth = 1
    
    line = ax.plot(dates, de_load_data, '-', dates, hertz_load_data,'-',
                   dates, amprion_load_data, '-', dates, tennet_load_data, '-',
                   dates, transnetbw_load_data, '-', linewidth=linewidth, label=de_load)
    #ax.plot(dates, hertz_load_data, '-', linewidth=linewidth, label=hertz_load)
    #ax.plot(dates, amprion_load_data, '-', linewidth=linewidth, label=amprion_load)
    #ax.plot(dates, tennet_load_data, '-', linewidth=linewidth, label=tennet_load)
    #ax.plot(dates, transnetbw_load_data, '-', linewidth=linewidth, label=transnetbw_load)
    
    ax.set_xlabel(time_col)
    ax.set_ylabel('load [MW]')
    ax2 = fig.add_subplot(212)
    ax2.axis("off") 
    #ax2.legend(pie[0],labels, loc="center")    
    plt.legend(line, [de_load, hertz_load, amprion_load, tennet_load, transnetbw_load], loc='center')#bbox_to_anchor=(0, 1.1), loc='upper left', ncol=1)
    plt.show()


def plot_map():
    power_df = pd.read_csv(data_path + "power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
    
    
time_col = 'cet_cest_timestamp'
de_load = 'DE_load_actual_entsoe_transparency'
hertz_load = 'DE_50hertz_load_actual_entsoe_transparency'
amprion_load = 'DE_amprion_load_actual_entsoe_transparency'
tennet_load = 'DE_tennet_load_actual_entsoe_transparency'
transnetbw_load = 'DE_transnetbw_load_actual_entsoe_transparency'

plot_all()
#plot_line()
#cleanfor_DE_load()
