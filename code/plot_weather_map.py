from glob_vars import data_path

from cartopy import crs as ccrs, feature as cfeat
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, ticker, cm
import pandas
import datetime
import xarray as xr
import numpy as np
import seaborn as sns


def plot_map_cartopy_netcdf():
    ds = xr.open_mfdataset(data_path + "netcdf_actuals/GridOneDayAhead_2017-06-01.nc")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # generate a basemap with country borders, oceans and coastlines
    ax.add_feature(cfeat.LAND)
    ax.add_feature(cfeat.OCEAN)
    ax.add_feature(cfeat.COASTLINE)
    ax.add_feature(cfeat.BORDERS, linestyle='dotted')

    area = ds.t2m - 273.15

    grid = area.sel(time='2017-06-01T12:00:00')
    grid.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='BuPu')
    #ax.imshow(grid, interpolation='bilinear')

    plt.show()

def plot_map_seaborn_csv():
    weather_df = pandas.read_csv(data_path + 'GridActuals_2017.csv', low_memory=False)
    weather_df = weather_df[weather_df['time'] == '2017-06-01 12:00:00']

    #weather_df.to_csv('/home/marcel/2017Grid.csv', ',')
    weather_df['t2m'] = weather_df['t2m'].apply(lambda x: x - 273.15)
    weather_df = weather_df[['latitude', 'longitude', 't2m']]

    x_set = sorted(set(weather_df['longitude']))
    y_set = sorted(set(weather_df['latitude']))
    x_size = len(x_set)
    y_size = len(y_set)

    sns.heatmap(np.array(weather_df['t2m']).reshape(y_size, x_size), cmap='BuPu')
    plt.show()


def plot_map_matplotlib_csv():
    # columns: latitude, longitude, time, tcc, u10, v10, t2m, lcc, tp
    weather_df = pandas.read_csv(data_path + 'GridActuals_2017.csv', low_memory=False)
    weather_df = weather_df[weather_df['time'] == '2017-06-01 12:00:00']

    #weather_df.to_csv('/home/marcel/2017Grid.csv', ',')
    weather_df['t2m'] = weather_df['t2m'].apply(lambda x: x - 273.15)

    x_set = sorted(set(weather_df['longitude']))
    y_set = sorted(set(weather_df['latitude']))
    x_size = len(x_set)
    y_size = len(y_set)
    #print(x_size)
    #print(y_size)

    #temp_list = np.reshape(list(weather_df['t2m']), (x_size, y_size))

    weather_df = weather_df[['latitude', 'longitude', 't2m']].pivot('latitude', 'longitude', 't2m')

    #fig = plt.figure()
    fig, ax = plt.subplots()

    #plt.plot(np.meshgrid(x_set, y_set))
    ax.imshow(weather_df, cmap='BuPu')#, interpolation='bilinear')
    ax.set_xticklabels(x_set)
    ax.set_yticklabels(y_set)
    ax.set_xticks(np.arange(len(x_set)))
    ax.set_yticks(np.arange(len(y_set)))
    # tick_spacing = 1
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #plt.colorbar(cm.ScalarMappable(cmap='BuPu'), ax=ax)
    plt.show()

data_path = data_path + 'ecmwf/'

plot_map_cartopy_netcdf()
plot_map_seaborn_csv()
plot_map_matplotlib_csv()
