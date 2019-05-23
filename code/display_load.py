from matplotlib import pyplot, dates
import csv
import pandas
import numpy

def cleanfor_DE_load():
    power_df = pandas.read_csv("/home/marcel/Repos/GISME/data/power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
    power_df = power_df[power_df[y_dim].notnull()]
    power_df[time_col] = pandas.to_datetime(power_df[time_col], format='%Y-%m-%dT%H:%M:%S%z')
    power_df[[time_col, y_dim]].to_csv('/home/marcel/test.csv', ',')

def plot_line():
    # utc_timestamp, cet_cest_timestamp
    power_df = pandas.read_csv("/home/marcel/Repos/GISME/data/power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
    power_df = power_df[power_df[y_dim].notnull()]
    power_df[time_col] = pandas.to_datetime(power_df[time_col], format='%Y-%m-%dT%H:%M:%S%z')
    dates = power_df[time_col]
    #print(dates.where(dates.notna()).head())
    data = power_df.where(power_df[y_dim].notna())[y_dim]
    pyplot.plot(dates, data, '-', linewidth=.5)
    pyplot.xlabel(time_col)
    pyplot.ylabel(y_dim)
    pyplot.show()


def plot_map():
    power_df = pandas.read_csv("/home/marcel/Projects/GISME/data/power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
    
    

time_col = 'cet_cest_timestamp'
y_dim = 'DE_load_entsoe_transparency'

plot_line()
