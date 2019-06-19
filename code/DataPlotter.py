#!/usr/bin/python3

from glob_vars import data_path, figure_path, lon_col, lat_col, de_load, bbox
from WeatherReader import WeatherReader
from LoadReader import LoadReader

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

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class DataPlotter:
    """
    
    """
    def __init__(self,fmt='pdf'):
        """Initializes WeatherPlot instance
        
        Parameters
        ----------
        reader : WeatherReader
                 used to retrieve weather data from nc files
        fmt    : string
                 format that figure is saved to
        
        Returns
        -------
        None
        """
        self.fmt = fmt
        self.wreader = WeatherReader()
        self.lreader = LoadReader()
    
    #def __create_fig_map(self,) # TODO write function to create figure from data, enables to make multiple subplots in one figure
    
    def __plot_days(self, days, fname,save=True,show=False):
        """Plot data for each day in days list and save file with specified format
        
        Parameters
        ----------
        days  : list
                list of days
        fname : string
                to name folder
        save  : boolean
                wether to save plot to file or not
        show  : boolean
                wether to show plot or not
        
        Returns
        -------
        None
        """
        var = days.name
        dir_pth = f'{figure_path}{var}/{fname}/'
        if not os.path.exists(dir_pth):
            os.makedirs(dir_pth)
        
        for day_num, day in enumerate(days["time"].values):
            data = self.wreader.vals4time(var, day)
            
            fig, ax = plt.subplots()
            
            cbox_bound = np.linspace(data.vmin,data.vmax,256)
            norm = colors.BoundaryNorm(cbox_bound, ncolors=256)
            img = ax.imshow(data.values, cmap='jet', extent=bbox, interpolation='bilinear', norm=norm)
            
            # read shapefile
            sf = shp.Reader('/home/marcel/Dropbox/data/shapes/DEU_adm1.shp')
            # iterate over all 16 states, in DEU_adm0.shp there is only one shape for germany
            for state in sf.shapes():
                points = np.array(state.points)
                intervals = list(state.parts) + [len(state.points)]
                ax.set_aspect(1)
                for (x, y) in zip(intervals[:-1 ], intervals[1:]):
                    ax.plot(*zip(*points[x:y]), color='k', linewidth=.4)
            
            cbox_ticks = np.linspace(data.vmin,data.vmax,8)
            cbar = fig.colorbar(img,ticks=cbox_ticks)
            cbar.set_label(self.wreader.get_long_name(var))
            ax.set_xlabel(lon_col)
            ax.set_title(f'date: {day}')
            ax.set_ylabel(lat_col)
            
            xticks = np.linspace(bbox[0],bbox[1],(bbox[1]-bbox[0])*4+1)
            ax.set_xticks(xticks,minor=True)
            yticks = np.linspace(bbox[2],bbox[3],(bbox[3]-bbox[2])*4+1)
            ax.set_yticks(yticks,minor=True)
            
            ax.grid(which='minor',alpha=0.2,color='k',linewidth=.5)
            ax.grid(which='major',alpha=0.4,color='k',linewidth=.5)
            
            file_name = f'{dir_pth}{day_num}_{pd.to_datetime(day).strftime("%Y%m%d%H")}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
            
            if save:
                print(f'saving plot for {day} in {file_name}')
                if type(self.fmt) is list:
                    for f in self.fmt:
                        fig.savefig(f'{file_name}.{f}', bbox_inches='tight', format=f, optimize=True, dpi=150)                    
                else:
                    fig.savefig(f'{file_name}.{self.fmt}', bbox_inches='tight', format=self.fmt, optimize=True, dpi=150)
            if show:
                plt.show()
            # close figures as they won't be closed automatically by python during runtime
            plt.close(fig)
    
    def plot_nmin(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the smallest values for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
        """
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        #print(self.wreader.nmin_val_days(var, n).name)
        #print(self.wreader.nmin_val_days(var, n).sizes)
        days = self.wreader.nmin_val_days(var, n)
        
        self.__plot_days(days, 'min', save, show)

    def plot_nmax(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the largest values for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
        """
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        days = self.wreader.nmax_val_days(var, n)
        
        self.__plot_days(days, 'max', save, show)
    
    def plot_nmin_var(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the smallest variance for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
        """
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        days = self.wreader.nminvar_val_days(var, n)
        
        self.__plot_days(days, 'minvar', save, show)
    
    def plot_nmax_var(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the largest variance for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
        """        
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        days = self.wreader.nmaxvar_val_days(var, n)
        
        self.__plot_days(days, 'maxvar', save, show)
    
    def plot_nmin_mean(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the smallest mean for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
        """        
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        days = self.wreader.nminmean_val_days(var, n, save, show)
        
        self.__plot_days(days, 'minmean', save, show)

    def plot_nmax_mean(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the largest mean for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
        """        
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        days = self.wreader.nmaxmean_val_days(var, n)
        
        self.__plot_days(days, 'maxmean', save, show)
    
    def plot_nmin_med(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the smallest median for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
        """        
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        days = self.wreader.nminmed_val_days(var, n)
        
        self.__plot_days(days, 'minmed', save, show)

    def plot_nmax_med(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the largest median for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
           """        
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        days = self.wreader.nmaxmed_val_days(var, n)
        
        self.__plot_days(days, 'maxmed', save, show)
    
    def plot_nmin_sum(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the smallest sum for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
        """        
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        days = self.wreader.nminsum_val_days(var, n)
        
        self.__plot_days(days, 'minsum', save, show)

    def plot_nmax_sum(self, var, n=4, save=True, show=False):
        """Plot/save the n days with the largest sum for the specified
           variable reduced over longitude and latitude
        
        Parameters
        ----------
        var : string
              name of variable
        n   : integer
              number of plots
        
        Returns
        -------
        None
        """        
        assert var in self.wreader.get_vars(), f"variable '{var}' not found"
        days = self.wreader.nmaxsum_val_days(var, n)
        
        self.__plot_days(days, 'maxsum', save, show)
    
    def plot_load_time_func(self,var,start,stop,func,load_col=de_load
                            ,freq=24,save=True,show=False):
        """Plot/save function of load and date with variable
           after applying given function to its data
        
        Parameters
        ----------
        var      : string
                   name of variable to plot
        start    : pandas.Timestamp
                   starting time (e.g. start = pandas.Timestamp(datetime(2015,1,1,12),tz='utc'))
        stop     : pandas.Timestamp
                   stopping time
        func     : function object
                   function applied to weather data to reduce over longitude and latitude
        load_col : string
                   specifies column in load file that will be plotted
        freq     : integer (where freq mod 2 == 0, as resolution of data is 2H)
                   specifies in what frequency of hours points will be plotted
        save     : boolean
                   wether to save plot to file or not
        show     : boolean
                   wether to show plot or not
        
        Returns
        -------
        None
        """
        assert (freq%2==0), "frequency must be dividable by 2 as resolution of data is 2h"
        
        fname = func.__name__
        
        rng = pd.date_range(start,stop,freq=f'{freq}H')
        load = self.lreader.vals4slice(load_col, start, stop, step=freq)
        ncval = self.wreader.reduce_lonlat(var, func).sel(time=rng)
        
        # select data for workdays
        ncweek = ncval.where((ncval['time.weekday'] != 5) & (ncval['time.weekday'] != 6), drop=True)
        loadweek = load.where((load['utc_timestamp.weekday'] != 5) & (load['utc_timestamp.weekday'] != 6), drop=True)
        rngweek = rng.where((rng.weekday != 5) & (rng.weekday != 6), other=pd.NaT).dropna()
        
        # select data for weekends
        ncwend = ncval.where((ncval['time.weekday'] == 5) | (ncval['time.weekday'] == 6), drop=True)
        loadwend = load.where((load['utc_timestamp.weekday'] == 5) | (load['utc_timestamp.weekday'] == 6), drop=True)
        rngwend = rng.where((rng.weekday == 5) | (rng.weekday == 6), other=pd.NaT).dropna()
        
        fig, ax = plt.subplots()
        ax.scatter(rngwend, loadwend, s=10, c=ncwend, cmap='jet', marker='>', label='weekend')
        ax.scatter(rngweek, loadweek, s=10, c=ncweek, cmap='jet', marker='<', label='workday')
        ax.set_ylabel(f'{load_col} (MW)')
        ax.set_xlabel('date')
        ax.legend()
        
        cmap = plt.get_cmap('jet',256)
        norm = colors.Normalize(vmin=ncval.min(),vmax=ncval.max())
        scal_map = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
        ticks = np.linspace(ncval.min(),ncval.max(),8)
        
        cbar = fig.colorbar(scal_map,ticks=ticks,ax=ax)
        cbar.ax.set_ylabel(f'{self.wreader.get_long_name(var)} {fname} reduce over DE (K)', rotation=-90, va="bottom")
        
        pth = figure_path + 'plot_load_time_func/'
        if not os.path.exists(pth):
            os.makedirs(pth)
        file_name = pth + f'{var}_{fname}_{start.strftime("%Y%m%d%H")}_{stop.strftime("%Y%m%d%H")}_{freq}F'
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        if save:
            fig.savefig(f'{file_name}.{self.fmt}', bbox_inches='tight', format=self.fmt, optimize=True, dpi=150)
        if show:
            plt.show()
        
        plt.close(fig)


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
    weather_df = weather_df[[lat_col, lon_col, 't2m']]

    x_set = sorted(set(weather_df[lon_col]))
    y_set = sorted(set(weather_df[lat_col]))
    x_size = len(x_set)
    y_size = len(y_set)

    sns.heatmap(np.array(weather_df['t2m']).reshape(y_size, x_size), cmap='BuPu')
    plt.show()


data_path = data_path + 'ecmwf/'

fmt='pdf'
var='t2m'
n=1
#variables=['u10', 'v10', 't2m', 'e', 'ie', 'kx', 'lcc', 'skt', 'str', 'sp', 'tcc', 'tcwv', 'tp']

# numpy functions
funcs = [np.nanmin,np.nanmax,np.nanvar,np.nanmean,np.nanmedian,np.nansum]

start = pd.Timestamp(2015,1,1,12)
stop = pd.Timestamp(2018,12,31,12)
freq = 24

pl = DataPlotter(fmt)

rd= WeatherReader()

#for var in rd.get_vars():
    #for func in funcs:
        #pl.plot_load_time_func(var,start,stop,func)

#pl.plot_load_time_func(var, start, stop, np.mean,show=True)

pl.plot_nmin(var,n,save=False, show=True)
#rd = WeatherReader()
#for var in rd.get_vars():
    #pl.plot_nmin(var,n)
    #pl.plot_nmax(var,n)
    #pl.plot_nmin_var(var,n)
    #pl.plot_nmax_var(var,n)
    #pl.plot_nmin_mean(var,n)
    #pl.plot_nmax_mean(var,n)
    #pl.plot_nmin_med(var,n)
    #pl.plot_nmax_med(var,n)
    #pl.plot_nmin_sum(var,n)
    #pl.plot_nmax_sum(var,n)

