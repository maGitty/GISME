#!/usr/bin/python3

from glob_vars import data_path, figure_path, lon, lat
from NC_Reader import NC_Reader

import pandas
import xarray as xr
import numpy as np
import seaborn as sns
import shapefile as shp
from datetime import datetime, time
from calendar import monthrange
from cartopy import crs as ccrs, feature as cfeat
from matplotlib import pyplot as plt, colors, ticker, cm


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


def plot_map_matplotlib_csv(date):
    val = 't2m'
    
    # columns: latitude, longitude, time, tcc, u10, v10, t2m, lcc, tp
    weather_df = pandas.read_csv(f'{data_path}GridActuals_{date.year}.csv', low_memory=False)
    weather_df = weather_df[weather_df['time'] == str(date)]

    #weather_df.to_csv('/home/marcel/2017Grid.csv', ',')
    weather_df[val] = weather_df[val].apply(lambda x: x - 273.15)

    x_set = sorted(set(weather_df[lon]))
    y_set = sorted(set(weather_df[lat]))
    x_size = len(x_set)
    y_size = len(y_set)

    weather_df = weather_df[[lat, lon, val]].pivot(index=lat, columns=lon, values=val)[::-1]

    fig, ax = plt.subplots()

    img = ax.imshow(weather_df, cmap='jet', extent=(6, 15, 47.5, 55), interpolation='bilinear')
    
    # plot map
    sf = shp.Reader('/home/marcel/Dropbox/data/DEU_adm1.shp')
    for i in range(16):
        shape = sf.shape(i)
        points = np.array(shape.points)
        
        intervals = list(shape.parts) + [len(shape.points)]
        
        #ax = plt.gca()
        ax.set_aspect(1)
        
        for (i, j) in zip(intervals[:-1], intervals[1:]):
            ax.plot(*zip(*points[i:j]), color='k', linewidth=.5)
    
    bounds = np.linspace(-20,40,60)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    #pcm = ax[0].pcolormesh()
    
    fig.colorbar(img)
    ax.set_xlabel(lon)
    ax.set_title(f'date: {date}')
    ax.set_ylabel(lat)
    plt.show()


def plot_NC_read(date, save=False):
    pth = f'{data_path}netcdf_actuals/GridOneDayAhead_{date.strftime("%Y-%m-%d")}.nc'
    reader = NC_Reader(pth)
    data, bbox, long_name = reader.var4time('t2m', date)
    data = data - 273.15

    fig, ax = plt.subplots()

    img = ax.imshow(data, cmap='jet', extent=bbox, interpolation='bilinear')
    
    # plot map
    sf = shp.Reader('/home/marcel/Dropbox/data/DEU_adm1.shp')
    for i in range(16):
        shape = sf.shape(i)
        points = np.array(shape.points)
        
        intervals = list(shape.parts) + [len(shape.points)]
        
        #ax = plt.gca()
        ax.set_aspect(1)
        
        for (i, j) in zip(intervals[:-1], intervals[1:]):
            ax.plot(*zip(*points[i:j]), color='k', linewidth=.4)
    
    bounds = np.linspace(-20,40,60)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    #pcm = ax[0].pcolormesh()
    
    fig.colorbar(img)
    ax.set_xlabel(lon)
    ax.set_title(f'date: {date}')
    ax.set_ylabel(lat)
    
    if save:
        fig.savefig(f'{figure_path}nc_plot_{date.strftime("%Y%m%d%H")}.eps', bbox_inches='tight', format='eps')
    
    plt.show()


def plot_highest_var():
    years = range(2017,2018)
    months = range(1,13)
    dates = [datetime(year,month,day,12) for year in years for month in months for day in range(1,monthrange(year, month)[1]+1)]
    
    wvars = []
    for year in years:
        weather_df = pd.read_csv(f'{data_path}GridActuals_2017.csv', low_memory=False)
        for date in dates:
            wvar = weather_df[weather_df['time'] == str(date)].loc[:, 't2m'].var()
            #print(wvar)
            wvars.append((date,wvar))
    
    wvars = sorted(wvars, key=lambda x: -x[1])
    for i in range(5):
        date = wvars[i][0]
        plot_map_matplotlib_csv(date)

data_path = data_path + 'ecmwf/'

#plot_map_cartopy_netcdf()
#plot_map_seaborn_csv()
#plot_map_matplotlib_csv(datetime(2018,6,1,12))
plot_NC_read(datetime(2018,6,1,12), save=True)