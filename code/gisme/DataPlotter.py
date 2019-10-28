#!/usr/bin/python3

"""
This module is supposed to take care of visualizational
concerns especially if it is about plotting data
"""

from gisme import (figure_path, lon_col, lat_col, de_load, bbox,
                   variable_dictionary, data_path, nuts3_01res_shape,
                   nuts0_shape, isin_path, log, demography_file)
from gisme.WeatherReader import WeatherReader
from gisme.LoadReader import LoadReader
from gisme.Predictions import ARMAXForecast
from gisme.Utility import Utility

import os
import itertools
import pandas as pd
import numpy as np
import shapefile as shp
from datetime import datetime,timedelta
from descartes import PolygonPatch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt, colors, cm, markers, rc, rcParams

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text',usetex=True)
# plt.rc('text',usetex=True)
# plt.rc('font',family='serif')


class DataPlotter:
    """
    handles plotting for load and weather data
    
    Attributes
    ----------
    fmt     : string
              preferred output format of plots such as 'pdf', 'eps', 'jpg' or
              similar supported by matplotlib
    save    : boolean
              whether to save plot or not
    show    : boolean
              whether to display plot or not
    shape   : integer tuple
              for map multiplot, specify arrangement as tuple of length 2
    isin    : boolean
              for map plot, whether to filter by isinDE mask
    wreader : WeatherReader
              used to load weather data
    lreader : LoadReader
              used to load load data
    """
    def __init__(self, fmt='pdf', save=True, show=False, shape=None, isin=False):
        """Initializes WeatherPlot instance
        
        Parameters
        ----------
        fmt    : string
                 format that figure is saved to
        save  : boolean
                whether to save plots to file or not
        show  : boolean
                whether to show plots or not
        shape : tuple
                if multiplot should be plotted,specify shape
        isin  : boolean
                whether to filter values from outside germany
        
        Returns
        -------
        None
        """
        self.fmt = fmt
        self.save = save
        self.show = show
        self.shape = shape
        self.isin = isin
        self.wreader = WeatherReader(self.isin)
        self.lreader = LoadReader()
        
        # #change font size depending on size of plot
        # if self.shape is not None:
        #     rcParams.update({'font.size': 18./np.round(np.sqrt(self.shape[0]*self.shape[1]))})
        # else:
        #     rcParams.update({'font.size': 18.})
    
    def __save_show_fig(self, fig, dir_pth, file_name):
        """Save and show the passed figure if specified and finally close it
        
        Parameters
        ----------
        fig       : matplotlib.figure.Figure
                    the created figure
        dir_pth   : string
                    the path to the directory to save to
        file_name : string
                    the file name to save to
        
        Returns
        -------
        None
        """
        if self.save:
            log.info(f'saving plot in {file_name}')
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
    
    def __create_ax_map(self, ax, variable, time, norm, xlbl_true=None, ylbl_true=None):
        """Plot a map of germany on the given axis
        
        Parameters
        ----------
        ax        : matplotlib.axes.Axes
                    the axis used to plot
        variable  : string
                    the name of the weather variable
        time      : datetime.datetime
                    the time for which to plot the variable
        norm      : matplotlib.colors.BoundaryNorm
                    the norm for color distribution
        xlbl_true : bool
                    wether to plot a label for the x-axis or not
        ylbl_true : bool
                    wether to plot a label for the y-axis or not
        
        Returns
        -------
        None
        """
        if self.isin:
            data = self.wreader.vals4time(variable, time, isin=True)
            data.plot.imshow(ax=ax, cmap='jet', extent=bbox, norm=norm, add_colorbar=False)
        else:
            data = self.wreader.vals4time(variable, time)
            ax.imshow(data.values, cmap='jet', extent=bbox, interpolation='bilinear', norm=norm)

        # read shapefile
        eu_shape = shp.Reader(nuts0_shape)
        de_shape = None
        for record in eu_shape.shapeRecords():
            if 'DE' in record.record:
                de_shape = record
                break
        if de_shape is None:
            raise Exception('shape for germany could not be found!')

        # concatenate points so that single lines can be drawn
        state = de_shape.shape
        points = np.array(state.points)
        intervals = list(state.parts) + [len(state.points)]
        for (x, y) in zip(intervals[:-1], intervals[1:]):
            ax.plot(*zip(*points[x:y]), color='k', linewidth=2)

        ax.set_title(pd.to_datetime(time).strftime("%Y/%m/%d %HH"))
        # print x and y label only if is most left/lowest plot
        if xlbl_true:
            ax.set_xlabel(lon_col)
        else:
            ax.set_xlabel(None)
        if ylbl_true:
            ax.set_ylabel(lat_col)
        else:
            ax.set_ylabel(None)
        
        # set ticks for x and y in order to display the grid
        xticks = np.linspace(bbox[0], bbox[1], (bbox[1] - bbox[0]) * 4 + 1)
        ax.set_xticks(xticks, minor=True)
        yticks = np.linspace(bbox[2], bbox[3], (bbox[3] - bbox[2]) * 4 + 1)
        ax.set_yticks(yticks, minor=True)
        
        # plot own grid
        xgrid = np.linspace(bbox[0]+.125, bbox[1] - .125, num=(bbox[1] - bbox[0]) * 4)
        ygrid = np.linspace(bbox[2]+.125, bbox[3] - .125, num=(bbox[3] - bbox[2]) * 4)
        for xpoint in xgrid:
            ax.axvline(xpoint, alpha=.2, color='k', linewidth=.5, linestyle='--')
        for ypoint in ygrid:
            ax.axhline(ypoint, alpha=.2, color='k', linewidth=.5, linestyle='--')

    def __plot_days(self, days, fname):
        """Plot data for each day in days list and save file with specified format
        
        about file name format for single days:
            the leading number indicates the position in terms of min/max
            --> for min,0 means it's the smallest value,
                for max,the highest number corresponds to the highest value
        
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
        
        vmin, vmax = self.wreader.get_minmax(var)
        norm = colors.BoundaryNorm(np.linspace(vmin, vmax, 256), ncolors=256)
        cbox_ticks = np.linspace(vmin, vmax, 8)
        smap = cm.ScalarMappable(norm=norm, cmap=cm_jet)
        
        if self.shape is not None:
            fig, axs = plt.subplots(*self.shape, constrained_layout=True)
            
            day_list = list(days['time'].values)
            
            for xcoord in range(0, self.shape[1]):
                for ycoord in range(0, self.shape[0]):
                    ax = axs[xcoord, ycoord]
                    day = day_list.pop()
                    self.__create_ax_map(ax, var, day, norm, xcoord == (self.shape[1] - 1), ycoord == 0)
            cbar = fig.colorbar(smap, ticks=cbox_ticks, ax=axs.ravel().tolist())
            cbar.set_label(self.wreader.get_long_name(var))
            
            dir_pth = os.path.join(figure_path, var, 'bundles')
            file_name = os.path.join(dir_pth, f'{fname}{len(days)}_maps{"_isin" if self.isin else ""}')
            
            self.__save_show_fig(fig, dir_pth, file_name)
        else:
            for day_num, day in enumerate(days["time"].values):
                fig, ax = plt.subplots()
                
                self.__create_ax_map(ax, var, day, norm, xlbl_true=True, ylbl_true=True)
    
                cbar = fig.colorbar(smap, ticks=cbox_ticks)
                cbar.set_label(self.wreader.get_long_name(var))
        
                dir_pth = os.path.join(figure_path, var, fname)
                file_name = os.path.join(dir_pth, f'{day_num}_map{"_isin" if self.isin else ""}')
                
                self.__save_show_fig(fig, dir_pth, file_name)
    
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
        """Plot map showing which grid points are within germany
        
        Returns
        -------
        None
        """
        try:
            contained = np.load(os.path.join(isin_path, 'isinDE.npy'))
        except:
            log.info(f'isin file not found in {data_path}')
            util = Utility()
            contained = util.check_isinDE()
        
        fig, ax = plt.subplots()
        ax.imshow(contained, cmap=plt.cm.Greys, extent=bbox)
        
        ax.set_ylabel(lat_col)
        ax.set_xlabel(lon_col)
        
        file_name = os.path.join(figure_path, 'isinDE')
        self.__save_show_fig(fig, figure_path, file_name)

    def plot_isin_region(self, region_id):
        """Plot map showing which grid points are within specified region

        Parameters
        ----------
        region_id : string
                    the id of the region as string

        Returns
        -------
        None
        """
        util = Utility()
        try:
            contained = np.load(os.path.join(isin_path, f'isin{region_id}.npy'))
        except:
            log.info(f'isin file not found in {isin_path} for region {region_id}')
            contained = util.check_isin_region(region_id)
        
        fig, ax = plt.subplots()
        ax.imshow(contained, cmap=plt.cm.Greys, extent=bbox)
        
        ax.set_ylabel(lat_col)
        ax.set_xlabel(lon_col)
        
        file_name = os.path.join(figure_path, f'isin{util.get_region_name(region_id)}_{region_id}')
        self.__save_show_fig(fig, figure_path, file_name)

    def plot_isin_top_n(self, n, year):
        """Plot map showing which grid points are within n regions with highest population for specified year

        Parameters
        ----------
        n    : integer
               number of regions with highest population to plot
        year : integer
               specifies for which year population is checked

        Returns
        -------
        None
        """
        util = Utility()
        contained = util.demo_top_n_regions_map(n,2018)
        
        fig, ax = plt.subplots()
        ax.imshow(contained, cmap=plt.cm.Greys, extent=bbox)
        
        ax.set_ylabel(lat_col)
        ax.set_xlabel(lon_col)
        
        file_name = os.path.join(figure_path, f'isin_top{n}_year{year}')
        self.__save_show_fig(fig, figure_path, file_name)

    def plot_demo4year(self, year):
        """Plot a map of germany showing regional population data on NUTS 3 level
        
        Parameters
        ----------
        year : int
               name of variable
        
        Returns
        -------
        None
        """
        assert (year >= 2015 and year <=2018), "demography data only existing from 2015 to 2018"
        demo_df = pd.read_csv(demography_file, encoding='latin1', index_col='GEO')
        demo_df['Value'] = demo_df['Value'].map(lambda val: pd.NaT if val == ':' else float(val.replace(',', '')))
        df = demo_df[demo_df['TIME'] == year]
        with shp.Reader(nuts3_01res_shape) as nuts3_sf:
            regions = [rec for rec in nuts3_sf.shapeRecords() if rec.record['CNTR_CODE'] == 'DE']        
        values = np.array([df.loc[region.record['NUTS_ID'], :]['Value'] for region in regions]) / 1000
        _min = values.min()
        _max = values.max()
        
        fig, ax = plt.subplots()
        plt.xlim([5.5, 15.5])
        plt.ylim([47, 55.5])
        ax.set_xlabel(lon_col)
        ax.set_ylabel(lat_col)
        
        # for logarithmic colorbar
        cbox_bound = np.exp(np.linspace(np.log(_min), np.log(_max), 256))
        norm = colors.BoundaryNorm(cbox_bound, ncolors=256)
        sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('jet'))
        cbar = plt.colorbar(sm)
        cbar.set_label('inhabitants (in 1k)')
        
        for value, region in zip(values, regions):
            ax.add_patch(PolygonPatch(region.shape.__geo_interface__, fc=sm.to_rgba(value), ec='none'))
        
        dir_pth = os.path.join(figure_path, 'demo')
        file_name = os.path.join(dir_pth, f'demo{year}_logscale')
        
        self.__save_show_fig(fig, dir_pth, file_name)
    
    def plot_load_acf(self, lags=48, hour_steps=1, ndiff=0):
        """Plot autocorrelation plot of load data within given time range
           and given lags, hour steps and number of differences

        Parameters
        ----------
        lags       : integer
                     specifies number of lags shown on plot
        hour_steps : integer
                     specifies time steps in data in hours
        ndiff      : integer
                     specifies how often to differentiate before plotting

        Returns
        -------
        None
        """
        data = self.lreader.vals4step(de_load, step=hour_steps).interpolate_na(dim='utc_timestamp', method='linear')\
                   .diff(dim='utc_timestamp', n=ndiff).values
        fig = plot_acf(data, fft=True, use_vlines=True, lags=lags)
        dir_pth = os.path.join(figure_path, 'ACF')
        file_name = os.path.join(dir_pth, f'load_{lags}lags_ndiff{ndiff}_hstep{hour_steps}')
        self.__save_show_fig(fig, dir_pth, file_name)
    
    def plot_load_pacf(self, lags=48, hour_steps=1, ndiff=0):
        """Plot partial autocorrelation plot of load data within given time
           range and given lags, hour steps and number of differneces

        Parameters
        ----------
        lags       : integer
                     specifies number of lags shown on plot
        hour_steps : integer
                     specifies time steps in data in hours
        ndiff      : integer
                     specifies how often to differentiate before plotting

        Returns
        -------
        None
        """
        data = self.lreader.vals4step(de_load, step=hour_steps).interpolate_na(dim='utc_timestamp', method='linear')\
                   .diff(dim='utc_timestamp', n=ndiff).values
        fig = plot_pacf(data, use_vlines=True, lags=lags)
        dir_pth = os.path.join(figure_path, 'PACF')
        file_name = os.path.join(dir_pth, f'load_{lags}lags_ndiff{ndiff}_hstep{hour_steps}')
        self.__save_show_fig(fig, dir_pth, file_name)
    
    def plot_load_time_func(self, var, start, stop, func, load_col=de_load,
                            freq=24, aspect=(12, 5), skip_bottom_labels=False):
        """Plot/save function of load and date with variable
           after applying given function to its data
        
        Parameters
        ----------
        var                : string
                             name of variable to plot
        start              : pandas.Timestamp
                             starting time (e.g. start = pandas.Timestamp(datetime(2015,1,1,12),tz='utc'))
        stop               : pandas.Timestamp
                             stopping time
        func               : function object
                             function applied to weather data to reduce over longitude and latitude
        load_col           : string
                             specifies column in load file that will be plotted
        freq               : integer (where freq mod 2 == 0, as resolution of data is 2H)
                             specifies in what frequency of hours points will be plotted
        aspect             : tuple of ints
                             defines the aspect ratio of the plot
        skip_bottom_labels : boolean
                             specifies whether to skip the bottom label or not
        
        Returns
        -------
        None
        """
        assert (freq % 2 == 0), "frequency must be dividable by 2 as resolution of data is 2h"
        
        fname = func.__name__
        
        rng = pd.date_range(start, stop, freq=f'{freq}H')
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
        fig, ax = plt.subplots(figsize=aspect)
        ax.scatter(rngwend, loadwend, s=24, c=ncwend, cmap='jet', edgecolors='none', label='weekend',
                   marker=markers.MarkerStyle(marker="^", fillstyle='none'))
        ax.scatter(rngweek, loadweek, s=12, c=ncweek, cmap='jet', edgecolors='none', label='workday')
        ax.set_ylabel(variable_dictionary[load_col])
        ax.legend()  # , markerscale=2)
        
        cmap = plt.get_cmap('jet', 256)
        norm = colors.Normalize(vmin=ncval.min(), vmax=ncval.max())
        scal_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        ticks = np.linspace(ncval.min(), ncval.max(), 8)
        
        cbar = fig.colorbar(scal_map, ticks=ticks, ax=ax)  # , pad=.04, shrink=.6) # shrink to fit if aspect is changed
        cbar.ax.set_ylabel(f'{self.wreader.get_long_name(var)} {fname} reduce over DE (K)',
                           rotation=90, rotation_mode='anchor')
        
        if not skip_bottom_labels:
            # rotate labels from x-axis by 30° for readability
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        else:
            ax.set_xticklabels([])
        
        dir_pth = os.path.join(figure_path, 'plot_load_time_func')
        file_name = os.path.join(dir_pth, f'{var}_{fname}_{aspect[0]}A{aspect[1]}_'
                                 f'{start.strftime("%Y%m%d%H")}_{stop.strftime("%Y%m%d%H")}_{freq}F')
        
        self.__save_show_fig(fig, dir_pth, file_name)
    
    def plot_load(self, var, start, stop, freq=1, aspect=(12, 5), skip_bottom_labels=False):
        """Plot/save function of load variable/s
        
        Parameters
        ----------
        var                : list of strings
                             names of variables to plot
        start              : pandas.Timestamp
                             starting time (e.g. start = pandas.Timestamp(datetime(2015,1,1,12),tz='utc'))
        stop               : pandas.Timestamp
                             stopping time
        freq               : integer (where freq mod 2 == 0, as resolution of data is 2H)
                             specifies in what frequency of hours points will be plotted
        aspect             : tuple of ints
                             defines the aspect ratio of the plot
        skip_bottom_labels : boolean
                             specifies whether to skip the bottom label or not
        
        Returns
        -------
        None
        """
        # assert var in self.lreader.get_vars(), f'variable "{var}" not found' # assert not needed
        
        fig, ax = plt.subplots(figsize=aspect)
        
        drange = pd.date_range(start, stop, freq=f'{freq}H')
        
        for var_name in var:
            data = self.lreader.vals4time(var_name, drange)
            ax.plot(drange, data, '-', linewidth=1, label=variable_dictionary[var_name])
        
        ax.set_xlabel('UTC time')
        ax.set_ylabel('load (MW)')
        ax.set_ylim(-2000, 120000)
        ax.legend(loc='upper right')
        
        if not skip_bottom_labels:
            # rotate labels from x-axis by 30° for readability
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        else:
            ax.set_xticklabels([])
            
        dir_pth = os.path.join(figure_path, 'load_plot')
        file_name = os.path.join(dir_pth, f'{"_".join([varname[:7] for varname in var])}_{aspect[0]}A{aspect[1]}_'
                                 f'{start.strftime("%Y%m%d%H")}_{stop.strftime("%Y%m%d%H")}_{freq}F')
        self.__save_show_fig(fig, dir_pth, file_name)
    
    def plot_armax_forecast(self, tstart, tstop, forecast_end, p, q, exog=None,
                            save_armax=False, plot_range=None):
        """Plot an ARMAX forecast for the given parameters
        
        Parameters
        ----------
        tstart       : datetime.datetime
                       start time
        tstop        : datetime.datetime
                       stop time
        forecast_end : datetime.datetime
                       stop time of forecast (=tstop + forecast length)
        p            : integer
                       specifies number of AR coefficients
        q            : integer
                       specifies number of MA coefficients
        exog         : list of strings
                       specifies the variables to include as exogenous variables
        save_armax   : bool
                       specifies whether to save the armax or not
        plot_range   : None or tuple of datetime
                       specifies wether to plot specific range of forecast
        
        Returns
        -------
        None
        """
        armax = ARMAXForecast(tstart, tstop, p, q, exog=exog, const=False)
        armax.train()
        armax.summary()
        if save_armax:
            armax.save()
        forecast = armax.predict_one_step_ahead(forecast_end)
        fig, ax = plt.subplots()
        # for i,hours in enumerate(hours_range):
        #     ax.plot(fc_range,forecast[i],label=f'{hours}H forecast')
        #     log.info(armax.forecasts[i])
        log.info(armax.forecasts[0])
        
        if plot_range is None:
            fc_range = pd.date_range(tstop+timedelta(hours=1), forecast_end, freq='1H')
            ax.plot(fc_range, forecast.forecast, label='1H forecast')
            ax.plot(fc_range, forecast.actual, label='actual value')
        else:
            fc_range = pd.date_range(plot_range[0], plot_range[1], freq='1H')
            df = forecast.sel(plot_range[0], plot_range[1])
            ax.plot(fc_range, df['forecast'].values, label='1H forecast')
            ax.plot(fc_range, df['actual'].values, label='actual value')
        
        ax.set_ylabel('load [MW]')
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.legend()
        
        dir_pth = os.path.join(figure_path, 'ARMAXfc')
        file_name = os.path.join(dir_pth,
                                 f'ARMAX_p{p}q{q}_data{tstart.year}to{tstop.year}_fcto{forecast_end.strftime("%Y%m%d%H")}'
                                 f'{"" if exog is None else "_" + exog[0] if len(exog) == 1 else "_" + "_".join(exog)}'
                                 f'{"" if plot_range is None else "_plot_range" + plot_range[0].strftime("%Y%m%d%H")}'
                                 f'{"" if plot_range is None else "_" + plot_range[1].strftime("%Y%m%d%H")}')
        self.__save_show_fig(fig, dir_pth, file_name)

