#!/usr/bin/python3

from glob_vars import data_path, figure_path, lon, lat
from NC_Reader import NC_Reader

import os
import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import shapefile as shp
from datetime import datetime, time
from calendar import monthrange
from cartopy import crs as ccrs, feature as cfeat
from matplotlib import pyplot as plt, colors, ticker, cm


class NCPlot:
    """
    
    """
    def __init__(self):
        pass
    
    def _plot_days(self, days, var, ncreader, fname, fmt):
        """Plot data for each day in days list and save file with specified format
        Parameters
        ----------
        days     : list
                   list of days
        var      : string
                   column name of variable to plot
        ncreader : NC_Reader
                   instance to get values for each day
        fname    : string
                   to name folder
        fmt      : string or list of strings
                   to specify which format to store the figure
        """
        fig_pth = f'{figure_path}{var}_{fname}/'
        
        if not os.path.exists(fig_pth):
            os.makedirs(fig_pth)

        for day_num in range(len(days)):
            day = days[day_num]
            data, bbox, long_name, minmax = ncreader.vals4time(var, day)
            
            fig, ax = plt.subplots()
            
            cbox_bound = np.linspace(minmax[0],minmax[1],256)
            norm = colors.BoundaryNorm(cbox_bound, ncolors=256)
            img = ax.imshow(data, cmap='jet', extent=bbox, interpolation='bilinear', norm=norm)
            
            # plot map
            sf = shp.Reader('/home/marcel/Dropbox/data/shapes/DEU_adm1.shp')
            # iterate over all 16 states, in DEU_adm0.shp there is only one shape for germany
            for state in range(16):
                shape = sf.shape(state)
                points = np.array(shape.points)
                intervals = list(shape.parts) + [len(shape.points)]
                ax.set_aspect(1)
                
                for (x, y) in zip(intervals[:-1], intervals[1:]):
                    ax.plot(*zip(*points[x:y]), color='k', linewidth=.4)
            
            cbox_ticks = np.linspace(minmax[0],minmax[1],8)
            cbar = fig.colorbar(img,ticks=cbox_ticks)
            cbar.set_label(long_name)
            ax.set_xlabel(lon)
            ax.set_title(f'date: {day}')
            ax.set_ylabel(lat)
            
            if type(fmt) is list:
                for f in fmt:
                    pth = f'{fig_pth}{day_num}nc_plot_{pd.to_datetime(day).strftime("%Y%m%d%H")}_{datetime.now().strftime("%Y%m%d%H%M%S")}.{f}'
                    fig.savefig(pth, bbox_inches='tight', format=f, optimize=True, dpi=150)                    
            else:
                pth = f'{fig_pth}{day_num}_{pd.to_datetime(day).strftime("%Y%m%d%H")}_{datetime.now().strftime("%Y%m%d%H%M%S")}.{fmt}'
                fig.savefig(pth, bbox_inches='tight', format=fmt, optimize=True, dpi=150)
            # close figures as they won't be closed automatically by python during runtime
            plt.close(fig)
    
    def plot_nmin(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nmin_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'min', fmt)

    def plot_nmax(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nmax_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'max', fmt)
    
    def plot_nmin_var(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nminvar_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'minvar', fmt)
    
    def plot_nmax_var(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nmaxvar_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'maxvar', fmt)
    
    def plot_nmin_mean(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nminmean_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'minmean', fmt)

    def plot_nmax_mean(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nmaxmean_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'maxmean', fmt)
    
    def plot_nmin_med(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nminmed_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'minmed', fmt)

    def plot_nmax_med(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nmaxmed_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'maxmed', fmt)
    
    def plot_nmin_sum(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nminsum_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'minsum', fmt)

    def plot_nmax_sum(self, var, fmt='eps', n=4):
        reader = NC_Reader()
        hv = reader.nmaxsum_val_days(var, n)['time'].values
        
        self._plot_days(hv, var, reader, 'maxsum', fmt)


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
    weather_df = pd.read_csv(data_path + 'GridActuals_2017.csv', low_memory=False)
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
    weather_df = pd.read_csv(f'{data_path}GridActuals_{date.year}.csv', low_memory=False)
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


def plot_highest_var(fmt='eps', n=4):
    reader = NC_Reader()
    hv = reader.nmaxvar_val_days('t2m', 10)['time'].values
    
    new_folder = datetime.now().strftime('%Y%m%d%H%M%S/')
    fig_pth = figure_path + new_folder
    
    os.makedirs(fig_pth)
        
    for i in range(len(hv)):
        day = hv[i]
        data, bbox, long_name = reader.vals4time('t2m', day)
        
        fig, ax = plt.subplots()
        img = ax.imshow(data, cmap='jet', extent=bbox, interpolation='bilinear')
        
        # plot map
        sf = shp.Reader('/home/marcel/Dropbox/data/shapes/DEU_adm1.shp')        
        for state in range(16):
            shape = sf.shape(state)
            points = np.array(shape.points)
            
            intervals = list(shape.parts) + [len(shape.points)]
            
            #ax = plt.gca()
            ax.set_aspect(1)
            
            for (x, y) in zip(intervals[:-1], intervals[1:]):
                ax.plot(*zip(*points[x:y]), color='k', linewidth=.4)
        
        bounds = np.linspace(-20,40,60)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        #pcm = ax[0].pcolormesh()
        
        fig.colorbar(img)
        ax.set_xlabel(lon)
        ax.set_title(f'date: {day}')
        ax.set_ylabel(lat)
        
        fig.savefig(f'{fig_pth}nc_plot_{pd.to_datetime(day).strftime("%Y%m%d%H")}.{fmt}', bbox_inches='tight', format=fmt)        

data_path = data_path + 'ecmwf/'

fmt='pdf'
var='t2m'
variables=['u10', 'v10', 't2m', 'e', 'ie', 'kx', 'lcc', 'skt', 'str', 'sp', 'tcc', 'tcwv', 'tp']

pl = NCPlot()
#for var in variables:
    #pl.plot_nmax_var(var, fmt, 1)
pl.plot_nmin(var,fmt)
#pl.plot_nmax(var,fmt)
#pl.plot_nmin_var(var,fmt)
#pl.plot_nmax_var(var,fmt)
#pl.plot_nmin_mean(var,fmt)
#pl.plot_nmax_mean(var,fmt)
#pl.plot_nmin_med(var,fmt)
#pl.plot_nmax_med(var,fmt)
#pl.plot_nmin_sum(var,fmt)
#pl.plot_nmax_sum(var,fmt)

#plot_highest_var('jpg')

#plot_map_cartopy_netcdf()
#plot_map_seaborn_csv()
#plot_map_matplotlib_csv(datetime(2018,6,1,12))
#plot_NC_read(datetime(2018,6,1,12), save=True)
