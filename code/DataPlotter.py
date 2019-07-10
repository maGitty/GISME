#!/usr/bin/python3

from glob_vars import (figure_path,lon_col,lat_col,de_load,hertz_load,amprion_load,
                       tennet_load,transnet_load,bbox,variable_dictionary,data_path)
from WeatherReader import WeatherReader
from LoadReader import LoadReader

import os
import math
import pandas as pd
import numpy as np
import shapefile as shp
from datetime import datetime, time
from matplotlib import pyplot as plt, colors, ticker, cm
#import seaborn as sns
#import xarray as xr
#from calendar import monthrange
#from cartopy import crs as ccrs, feature as cfeat

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class DataPlotter:
    """
    handles plotting for load and weather data
    """
    def __init__(self,fmt='pdf',save=True,show=False,shape=None):
        """Initializes WeatherPlot instance
        
        Parameters
        ----------
        fmt    : string
                 format that figure is saved to
        save  : boolean
                wether to save plots to file or not
        show  : boolean
                wether to show plots or not
        
        Returns
        -------
        None
        """
        self.fmt = fmt
        self.wreader = WeatherReader()
        self.lreader = LoadReader()
        self.save = save
        self.show = show
        self.shape = shape
    
    def __create_fig_map(self,ax,variable,day,norm,xlbl_true=None,ylbl_true=None):
        """TODO
        
        """
        data = self.wreader.vals4time(variable, day)
        
        img = ax.imshow(data.values, cmap='jet', extent=bbox, interpolation='bilinear', norm=norm)
        
        # read shapefile
        sf = shp.Reader('/home/marcel/Dropbox/data/shapes/DEU_adm1.shp')
        # iterate over all 16 states, in DEU_adm0.shp there is only one shape for germany
        for state in sf.shapes():
            points = np.array(state.points)
            intervals = list(state.parts) + [len(state.points)]
            ax.set_aspect(1)
            for (x, y) in zip(intervals[:-1], intervals[1:]):
                ax.plot(*zip(*points[x:y]), color='k', linewidth=.4)
        ax.set_title(pd.to_datetime(day).strftime("%Y/%m/%d %HH"))
        # print x and y label only if is most left/lowest plot
        if xlbl_true:
            ax.set_xlabel(lon_col)
        if ylbl_true:
            ax.set_ylabel(lat_col)
        
        # set ticks for x and y in order to display the grid
        xticks = np.linspace(bbox[0],bbox[1],(bbox[1]-bbox[0])*4+1)
        ax.set_xticks(xticks,minor=True)
        yticks = np.linspace(bbox[2],bbox[3],(bbox[3]-bbox[2])*4+1)
        ax.set_yticks(yticks,minor=True)
        
        ax.grid(which='minor',alpha=0.2,color='k',linewidth=.5,linestyle='--')
        ax.grid(which='major',alpha=0.4,color='k',linewidth=.5,linestyle='--')
    
    def __plot_days(self, days, fname):
        """Plot data for each day in days list and save file with specified format
        
        about file name format for single days:
            the leading number indicates the position in terms of min/max
            --> for min, 0 means it's the smallest value,
                for max, the highest number corresponds to the highest value
        
        Parameters
        ----------
        days  : list
                list of days
        fname : string
                to name folder
        
        Returns
        -------
        None
        """
        assert (self.shape is None or self.shape[0]*self.shape[1] == len(days)),\
               "shape of plot must fit with number of plots"
        
        var = days.name
        cm_jet = cm.get_cmap('jet')
        
        vmin,vmax = self.wreader.get_minmax(var)
        norm = colors.BoundaryNorm(np.linspace(vmin,vmax,256), ncolors=256)
        cbox_ticks = np.linspace(vmin,vmax,8)
        smap = cm.ScalarMappable(norm=norm,cmap=cm_jet)
        
        if self.shape is not None:
            fig,axs = plt.subplots(*self.shape, constrained_layout=True)
            
            day_list = list(days['time'].values)
            
            for xcoord in range(0,self.shape[1]):
                for ycoord in range(0,self.shape[0]):
                    ax = axs[xcoord,ycoord]
                    day = day_list.pop()
                    self.__create_fig_map(ax, var, day,norm,xcoord==(self.shape[1]-1),ycoord==0)
            cbar = fig.colorbar(smap, ticks=cbox_ticks, ax=axs.ravel().tolist())
            cbar.set_label(self.wreader.get_long_name(var))
            
            if self.save:
                dir_pth = f'{figure_path}{var}/bundles/'
                file_name = f'{dir_pth}{fname}{len(days)}_maps'
                print(f'saving plot for {day} in {file_name}')
                if not os.path.exists(dir_pth):
                    os.makedirs(dir_pth)            
                
                if type(self.fmt) is list:
                    for f in self.fmt:
                        fig.savefig(f'{file_name}.{f}', bbox_inches='tight', format=f, optimize=True, dpi=150)                    
                else:
                    fig.savefig(f'{file_name}.{self.fmt}', bbox_inches='tight', format=self.fmt, optimize=True, dpi=150)
            if self.show:
                plt.show()
            # close figures as they won't be closed automatically by python during runtime
            plt.close(fig)            
        else:
            for day_num, day in enumerate(days["time"].values):
                fig, ax = plt.subplots()
                
                self.__create_fig_map(ax, var, day, norm, xlbl_true=True, ylbl_true=True)
    
                cbar = fig.colorbar(smap,ticks=cbox_ticks)
                cbar.set_label(self.wreader.get_long_name(var))
        
                if self.save:
                    dir_pth = f'{figure_path}{var}/{fname}/'
                    file_name = f'{dir_pth}{day_num}_map'
                    print(f'saving plot for {day} in {file_name}')
                    if not os.path.exists(dir_pth):
                        os.makedirs(dir_pth)
                    if type(self.fmt) is list:
                        for f in self.fmt:
                            fig.savefig(f'{file_name}.{f}', bbox_inches='tight', format=f, optimize=True, dpi=150)
                    else:
                        fig.savefig(f'{file_name}.{self.fmt}', bbox_inches='tight', format=self.fmt, optimize=True, dpi=150)
                if self.show:
                    plt.show()
                # close figures as they won't be closed automatically by python during runtime
                plt.close(fig)
    
    def plot_nmin(self, var, n=4):
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

        days = self.wreader.nmin_val_days(var, n)
        self.__plot_days(days, 'min')

    def plot_nmax(self, var, n=4):
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
        self.__plot_days(days, 'max')
    
    def plot_nmin_var(self, var, n=4):
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
        self.__plot_days(days, 'minvar')
    
    def plot_nmax_var(self, var, n=4):
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
        self.__plot_days(days, 'maxvar')
    
    def plot_nmin_mean(self, var, n=4):
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
        
        days = self.wreader.nminmean_val_days(var, n)
        self.__plot_days(days, 'minmean')

    def plot_nmax_mean(self, var, n=4):
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
        self.__plot_days(days, 'maxmean')
    
    def plot_nmin_med(self, var, n=4):
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
        self.__plot_days(days, 'minmed')

    def plot_nmax_med(self, var, n=4):
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
        self.__plot_days(days, 'maxmed')
    
    def plot_nmin_sum(self, var, n=4):
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
        self.__plot_days(days, 'minsum')

    def plot_nmax_sum(self, var, n=4):
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
        assert var in self.wreader.get_vars(), f'variable "{var}" not found'
        
        days = self.wreader.nmaxsum_val_days(var, n)
        self.__plot_days(days, 'maxsum')
    
    def plot_isin(self):
        """TODO
        
        """
        try:
            contained = np.load(f'{data_path}isin.npy')
        except:
            contained = self.wreader.check_isinDE()
        
        fig,ax = plt.subplots()
        ax.imshow(contained,cmap=plt.cm.gray, extent=bbox)
        
        if self.save:
            file_name = figure_path + 'isin'
            print(f'saving plot in {file_name}')
            fig.savefig(f'{file_name}.{self.fmt}', bbox_inches='tight', format=self.fmt, optimize=True, dpi=150)
        if self.show:
            plt.show()
        # close figures as they won't be closed automatically by python during runtime
        plt.close(fig)        
    
    def plot_load_time_func(self, var, start, stop, func, load_col=de_load, freq=24):
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
        
        # use figsize to stretch
        fig, ax = plt.subplots(figsize=(12,5))
        ax.scatter(rngwend, loadwend, s=8, c=ncwend, cmap='jet', edgecolors='none', label='weekend', marker="^", facecolors='none')
        ax.scatter(rngweek, loadweek, s=6, c=ncweek, cmap='jet', edgecolors='none', label='workday')
        ax.set_ylabel(variable_dictionary[load_col])
        ax.legend(markerscale=2)
        
        cmap = plt.get_cmap('jet',256)
        norm = colors.Normalize(vmin=ncval.min(),vmax=ncval.max())
        scal_map = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
        ticks = np.linspace(ncval.min(),ncval.max(),8)
        
        cbar = fig.colorbar(scal_map,ticks=ticks,ax=ax)#,pad=.04,shrink=.6) # shrink to fit if aspect is changed
        cbar.ax.set_ylabel(f'{self.wreader.get_long_name(var)} {fname} reduce over DE (K)', rotation=90, rotation_mode='anchor')
        
        # rotate labels from x-axis by 30° for readability
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        
        if self.save:
            pth = figure_path + 'plot_load_time_func/'
            file_name = pth + f'{var}_{fname}_{start.strftime("%Y%m%d%H")}_{stop.strftime("%Y%m%d%H")}_{freq}F'
            print(f'saving plot in {file_name}')
            if not os.path.exists(pth):
                os.makedirs(pth)
            fig.savefig(f'{file_name}.{self.fmt}', bbox_inches='tight', format=self.fmt, optimize=True, dpi=150)
        if self.show:
            plt.show()
        # close figures as they won't be closed automatically by python during runtime
        plt.close(fig)
    
    def plot_load(self, var, start, stop,freq):
        """Plot/save function of load variable/s
        
        Parameters
        ----------
        var      : list of strings
                   names of variables to plot
        start    : pandas.Timestamp
                   starting time (e.g. start = pandas.Timestamp(datetime(2015,1,1,12),tz='utc'))
        stop     : pandas.Timestamp
                   stopping time
        freq     : integer (where freq mod 2 == 0, as resolution of data is 2H)
                   specifies in what frequency of hours points will be plotted
        
        Returns
        -------
        None
        """
        #assert var in self.lreader.get_vars(), f'variable "{var}" not found' # assert not needed 
        
        fig, ax = plt.subplots(figsize=(12,5))
        
        drange = pd.date_range(start,stop,freq=f'{freq}H')
        
        for var_name in var:
            data = self.lreader.vals4time(var_name, drange)
            ax.plot(drange, data, '-', linewidth=1, label=variable_dictionary[var_name])
        
        ax.set_xlabel('UTC time')
        ax.set_ylabel('load (MW)')
        ax.set_ylim(-2000, 120000)
        ax.legend(loc='upper right')
        
        # rotate labels from x-axis by 30° for readability
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        
        if self.save:
            pth = figure_path + 'load_plot/'
            file_name = pth + f'{"_".join([varname[:7] for varname in var])}_{start.strftime("%Y%m%d%H")}_{stop.strftime("%Y%m%d%H")}_{freq}F'
            print(f'saving plot in {file_name}')
            if not os.path.exists(pth):
                os.makedirs(pth)
            fig.savefig(f'{file_name}.{self.fmt}', bbox_inches='tight', format=self.fmt, optimize=True, dpi=150)
        if self.show:
            plt.show()
        # close figures as they won't be closed automatically by python during runtime
        plt.close(fig)


fmt='pdf'
var='t2m'
n=1

# used numpy functions
funcs = [np.nanmin,np.nanmax,np.nanvar,np.nanmean,np.nanmedian,np.nansum]

start = pd.Timestamp(2015,1,1,12)
stop = pd.Timestamp(2018,12,31,12)
freq = 24

pl = DataPlotter(fmt,save=True,show=True)#,shape=(2,2))

rd= WeatherReader()
pl.plot_load([de_load,hertz_load,amprion_load,tennet_load,transnet_load], start, stop, freq)
#for var in rd.get_vars():
    #for func in funcs:
        #pl.plot_load_time_func(var,start,stop,func)

#pl.plot_load_time_func(var, start, stop, np.mean)

#pl.plot_nmax_var(var,4)
#pl.plot_isin()
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

