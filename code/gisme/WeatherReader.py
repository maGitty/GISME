#!/usr/bin/python3

"""
This module provides access to the used weather data as well as
several functions to filter,reduce or reorganize the data
"""

from gisme import (lon_col, lat_col, era5_path, data_path, isin_path, log)
from gisme.Utility import Utility

import os
import re
import numpy as np
import xarray as xr


class WeatherReader:
    """Used to read weather nc files and to return xarray.DataArray containing desired data
    
    Attributes
    ----------
    isin        : boolean
                  whether to filter data by isinDE mask
    util        : Utility
                  used for several utility functions
    filename    : string
                  files containing weather data
    wdata       : xarray.DataSet
                  the complete weather data
    var_names   : list of strings
                  short names of all variables
    date_bounds : tuple of datetime.datetime
                  start and end datetime of data
    """
    def __init__(self, isin=False):
        """Initializes instance, set path to open files, store some information for faster response

        Parameters
        ----------
        isin : boolean
               specifies whether to only return points within a shape or not

        Returns
        -------
        None
        """
        assert os.path.exists(era5_path), 'path to weather data does not exist'
        
        self.isin = isin
        self.util = Utility()
        self.filename = os.path.join(era5_path, '*.nc')
        with xr.open_mfdataset(self.filename) as nc_file:
            # dropna drops times with no values, carefully use, might throw away single points somewhere
            self.wdata = nc_file  # .dropna('time')
        
        self.var_names = [name for name in self.wdata.data_vars]
        self.date_bounds = self.wdata['time'].min().values, self.wdata['time'].max().values
    
    def _nminmax_reduce_days(self, name, func, minmax, n):
        """Private method, return list of n days with min/max values for variable
        
        Function is applied to reduce the data along longitude and latitude axes
        
        Parameters
        ----------
        name   : string
                 name of the variable
        func   : numpy ndarray function
                 function to apply along longitude/latitude
        minmax : string
                 specifies if n min or max must be returned
        n      : int
                 number of min/max values
              
        Returns
        -------
        xarray.DataArray of n days with min/max values for specified variable after applying func
        """
        assert (name in self.var_names and minmax in ['min', 'max']),\
            f'wrong variable name ({name}) or minmax not in ["min","max"]'
        
        data = self.reduce_lonlat(name, func).dropna('time')
        data = data.sortby(data)
        n_minmax = data[:n] if minmax is 'min' else data[-n:]
        
        return n_minmax
    
    def _nmin_reduce_days(self, name, func, n):
        """Private method, return list of n days with min values for variable
        
        Function is applied to reduce the data along longitude and latitude axes
           
        Parameters
        ----------
        name  : string
                name of the variable
        func  : numpy function
                a numpy function
        n     : integer
                number of days to return
        
        Returns
        -------
        xarray.DataArray with n values reduced along longitude and latitude
        """
        return self._nminmax_reduce_days(name, func, 'min', n)
    
    def _nmax_reduce_days(self, name, func, n):
        """Private method, return list of n days with max values for variable
        
        Function is applied to reduce the data along longitude and latitude axes
           
        Parameters
        ----------
        name  : string
                name of the variable
        func  : numpy function
                a numpy function
        n     : integer
                number of days to return
        
        Returns
        -------
        xarray.DataArray with n values reduced along longitude and latitude
        """
        return self._nminmax_reduce_days(name, func, 'max', n)
    
    def reduce_lonlat(self, name, func):
        """Return data for specified variable after applying function to reduce along longitude and latitude axes
           
        Parameters
        ----------
        name  : string
                name of the variable
        func  : numpy function
                a numpy function
        
        Returns
        -------
        xarray.DataArray with values reduced along longitude and latitude
        """
        assert name in self.var_names, f'column {name} not found'
        
        if self.isin:
            try:
                contained = np.load(os.path.join(isin_path, 'isinDE.npy'))
            except:
                log.info(f'isin file not found in {data_path}, creating new')
                contained = self.util.check_isinDE()
            return self.wdata[name].where(contained, other=np.nan, drop=False)\
                                   .reduce(func, dim=[lon_col, lat_col])
        else:
            return self.wdata[name].reduce(func, dim=[lon_col, lat_col])
    
    def reduce_lonlat_slice(self, name, func, start, stop):
        """Return data for specified variable
        
        Date is filtered by time slice and function is applied
        to reduce it along longitude and latitude axes
           
        Parameters
        ----------
        name  : string
                name of the variable
        func  : numpy function
                a numpy function
        start : datetime.datetime
                start time for values to be returned
        stop  : datetime.datetime
                stop time for values to be returned
        
        Returns
        -------
        xarray.DataArray with values reduced along longitude and latitude for time slice
        """
        assert name in self.var_names, f'column {name} not found'
        
        data = self.wdata[name].sel(time=slice(start, stop))
        
        if self.isin:
            try:
                contained = np.load(os.path.join(isin_path, 'isinDE.npy'))
            except:
                log.info(f'isin file not found in {data_path}, creating new')
                contained = self.util.check_isinDE()
            data = data.where(contained, other=np.nan, drop=False)
        return data.reduce(func, dim=[lon_col, lat_col])
    
    def get_size(self):
        """Returns shape of whole data as tuple"""
        return self.wdata.sizes
    
    def get_coords(self):
        """Returns coordinating dimensions and respective value lists"""
        return self.wdata.coords
    
    def get_vars(self):
        """Returns list of variable names held by data as list"""
        return self.var_names
    
    def get_date_bounds(self):
        """Returns tuple of upper/lower date boundaries"""
        return self.date_bounds
    
    def longitudes(self):
        """Returns numpy array of longitude values"""
        return self.wdata[lon_col].values
    
    def latitudes(self):
        """Returns numpy array of latitude values"""
        return self.wdata[lat_col].values
    
    def get_long_name(self, var):
        """Returns long name for given abreviated variable name
        
        Parameters
        ----------
        var : string
              short name of variable
        
        Returns
        -------
        long name of variable as string
        """
        return self.wdata[var].long_name
    
    def get_minmax(self, name):
        """Returns min and max for specified variable and times
        
        Parameters
        ----------
        name : string
               short name of variable
        days : string or some time format supported by xarray
               days for which min and max has to be returned
        
        Returns
        -------
        tuple of min and max
        """
        return np.floor(self.wdata[name].min().values), np.ceil(self.wdata[name].max().values)
    
    def print_vars_texfmt(self):
        """Prints all variables in format 'name & unit & min & max' to just insert to latex
        
        Returns
        -------
        None
        """
        pow_regex = re.compile('(\*\*)(\S+)')
        for var in self.var_names:
            variable = self.wdata[var].dropna('time')
            unit = pow_regex.sub("^{\g<2>}", variable.units).replace(' ', '~')
            print(f'{variable.long_name} & ${unit}$ & '
                  f'{variable.values.mean().round(2):.2f} & '
                  f'{variable.values.min().round(2):.2f} & '\
                  f'{variable.values.max().round(2):.2f}\\\\')
    
    def vals4time(self, name, datetime, isin=False):
        """Returns the values for specified variable and time
        
        Parameters
        ----------
        name     : string
                   name of variable in nc file that should be contained in df
        datetime : datetime.datetime
                   the specified datetime for which the data is returned
        isin     : bool
                   whether to only get points within germany or all
        
        Returns
        -------
        xarray.DataArray :
            2D data with values for variable over latitude and longitude
            long name of variable
        
        ready for plotting with imshow
        """
        assert name in self.var_names, f'column {name} not found'

        # filter data by name and time
        data = self.wdata[name].sel(time=datetime)
        
        if isin:
            try:
                contained = np.load(os.path.join(isin_path, 'isinDE.npy'))
            except:
                log.info(f'isin file not found in {data_path}')
                contained = self.util.check_isinDE()
            data = data.where(contained, other=np.nan, drop=False)
        
        # update DataArray attributes to add min and max values for complete time
        data.attrs.update({'vmin': np.floor(self.wdata[name].min().values), 'vmax': np.ceil(self.wdata[name].max().values)})
        
        return data

    def val4postime(self, name, lon, lat, dtime):
        """Returns value for variable at geographic position at specific time
        
        Parameters
        ----------
        name  : string
                name of the variable
        lon   : numeric
                the geographic longitude check getLon() for valid values
        lat   : numeric
                the geographic latitude check getLat() for valid values
        dtime : datetime.datetime
                the specified datetime for which the data is returned
        
        Returns
        -------
        single value for specified variable, position and time
        """
        assert name in self.var_names, f'column {name} not found'
        
        try:
            data = self.wdata[name].sel(longitude=lon, latitude=lat, time=dtime)
        except:
            raise Exception(f'coordinates not within bbox? lon:{lon}, lat:{lat} '
                            f'or specified time not found? time:{dtime}')
        return data
    
    def vals4lon_daytime(self, name, longitude, daytime):
        """Returns values for variable at geographic longitude at daytime averaged over all days
        
        Parameters
        ----------
        name      : string
                    name of the variable
        longitude : numeric
                    the geographic longitude check getLon() for valid values
        daytime   : datetime.datetime
                    the specified daytime for which the data is returned
        
        Returns
        -------
        values for specified variable and longitude averaged along latitude and daytime
        over all days as xarray.DataArray and the variables long name
        """
        assert name in self.var_names, f'column {name} not found'

        try:
            long_name = f'{self.wdata[name].long_name} ({self.wdata[name].units})'
            # get fields for variable at longitude and daytime averaged over all days
            data = self.wdata[name].sel(longitude=longitude, time=daytime).values.mean(axis=0)
        except:
            raise Exception(f'coordinates not within bbox? lon:{longitude}')
        return data, long_name
    
    def vals4lat_daytime(self, name, latitude, daytime):
        """Returns values for variable at geographic latitude at daytime averaged over all days
        
        Parameters
        ----------
        name     : string
                   name of the variable
        latitude : numeric
                   the geographic latitude check getLat() for valid values
        daytime  : datetime.datetime
                   the specified daytime for which the data is returned
        
        Returns
        -------
        values for specified variable and latitude averaged along longitude and daytime
        over all days as xarray.DataArray and the variables long name
        """
        assert name in self.var_names, f'column {name} not found'

        try:
            long_name = f'{self.wdata[name].long_name} ({self.wdata[name].units})'
            # get fields for variable at latitude and daytime averaged over all days
            data = self.wdata[name].sel(latitude=latitude, time=daytime).mean(dim='time')
        except:
            raise Exception(f'coordinates not within bbox? lat:{latitude}')
        return data, long_name

    def vals4timeslice_reduced(self,name,start,stop,func):
        """Return values for specified variable with one value per step start to stop reduced by given function
        
        Parameters
        ----------
        name  : string
                name of the variable
        start : datetime.datetime
                start time
        stop  : datetime.datetime
                stop time
        func  : numpy function
                function used to reduce along longitude and latitude
        
        Returns
        -------
        xarray.DataArray of specified variable reduced over longitude and latitude by max
        """
        return self.reduce_lonlat_slice(name, func, start, stop)

    def maxvals4timeslice(self, name, start, stop):
        """Return values for specified variable with one value per step start to stop reduced by max
        
        Parameters
        ----------
        name  : string
                name of the variable
        start : datetime.datetime
                start time
        stop  : datetime.datetime
                stop time
        
        Returns
        -------
        xarray.DataArray of specified variable reduced over longitude and latitude by max
        """
        return self.reduce_lonlat_slice(name, np.nanmax, start, stop)
    
    def meanvals4timeslice(self, name, start, stop):
        """Return values for specified variable with one value per step from start to stop reduced by mean
        
        Parameters
        ----------
        name  : string
                name of the variable
        start : datetime.datetime
                start time
        stop  : datetime.datetime
                stop time
        
        Returns
        -------
        xarray.DataArray of specified variable reduced over longitude and latitude by mean
        """
        return self.reduce_lonlat_slice(name, np.nanmean, start, stop)
    
    def stackedvals4timeslice(self, name, start, stop):
        """Return values for specified variable from start to stop
        
        The values are flattened by concatenating time steps of all grid points,
        so for a grid of size (x,y) with n steps, a numpy.ndarray
        with shape (x*y, n) will be returned
        
        Parameters
        ----------
        name  : string
                name of the variable
        start : datetime.datetime
                start time
        stop  : datetime.datetime
                stop time
        
        Returns
        -------
        xarray.DataArray with stacked grid points of specified variable
        """
        return self.wdata[name].sel(time=slice(start, stop)).stack(loc=(lon_col, lat_col)).transpose()
    
    def isin4timesliceDE(self, name, start, stop):
        """Return values for specified variable from start to stop filtered by DE map
        
        The values are flattened by concatenating time steps of all grid points
        
        Parameters
        ----------
        name  : string
                name of the variable
        start : datetime.datetime
                start time
        stop  : datetime.datetime
                stop time
        
        Returns
        -------
        xarray.DataArray with stacked grid points of specified variable filtered by DE map
        """
        try:
            contained = np.load(os.path.join(isin_path, 'isinDE.npy'))
        except:
            log.info(f'isin file not found in {data_path}')
            contained = self.util.check_isinDE()
        return self.wdata[name].sel(time=slice(start, stop)).where(contained, other=np.nan, drop=False)\
                   .stack(loc=(lon_col, lat_col)).dropna('loc').transpose().values
    
    def isin4timeslice_region(self, name, start, stop, region_id):
        """Return values for specified variable from start to stop filtered by region map
        
        The values are flattened by concatenating time steps of all grid points
        
        Parameters
        ----------
        name      : string
                    name of the variable
        start     : datetime.datetime
                    start time
        stop      : datetime.datetime
                    stop time
        region_id : string
                    id of region
        
        Returns
        -------
        xarray.DataArray with stacked grid points of specified variable filtered by region map
        """
        try:
            contained = np.load(os.path.join(isin_path, 'isinDE.npy'))
        except:
            log.info(f'isin file not found in {data_path}')
            contained = self.util.check_isin_region(region_id)
        return self.wdata[name].sel(time=slice(start, stop)).where(contained, other=np.nan, drop=False)\
                   .stack(loc=(lon_col, lat_col)).dropna('loc').transpose().values
    
    def isin4timeslice_map(self, name, start, stop, matrix):
        """Return values for specified variable from start to stop filtered by specified map
        
        The values are flattened by concatenating time steps of all grid points
        
        Parameters
        ----------
        name   : string
                 name of the variable
        start  : datetime.datetime
                 start time
        stop   : datetime.datetime
                 stop time
        matrix : 2 dimensional numpy.ndarray
                 mask to filter wanted grid points
        
        Returns
        -------
        xarray.DataArray with stacked grid points of specified variable filtered by specified map
        """
        return self.wdata[name].sel(time=slice(start, stop)).where(matrix, other=np.nan, drop=False)\
                   .stack(loc=(lon_col, lat_col)).dropna('loc').transpose()
    
    def demography_top_n_regions4timeslice(self, name, start, stop, n, year):
        """Return values for specified variable from start to stop filtered by top n regions map
        
        The values are flattened by concatenating time steps of all grid points
        
        Parameters
        ----------
        name  : string
                name of the variable
        start : datetime.datetime
                start time
        stop  : datetime.datetime
                stop time
        n     : integer
                number of regions with highest population
        year  : integer
                year for which to check population
        
        Returns
        -------
        xarray.DataArray with stacked grid points of specified variable filtered by top n regions map
        """
        return self.isin4timeslice_map(name, start, stop, self.util.demo_top_n_regions_map(n, year))
    
    def var_over_time(self, name):
        """Returns variance over time reduced along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        
        Returns
        -------
        xarray.DataArray for variable for each day reduced over longitude and latitude
        """
        assert name in self.var_names, f'column {name} not found'
        
        return self.reduce_lonlat(name, np.nanvar)
    
    def nmin_val_days(self, name, n=4):
        """Returns n min value days reduced with np.min along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmin_reduce_days(name, np.nanmin, n)

    def nmax_val_days(self, name, n=4):
        """Returns n max value days reduced with np.min along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmax_reduce_days(name, np.nanmin, n)
    
    def nminvar_val_days(self, name, n=4):
        """Returns n min value days reduced with np.var along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmin_reduce_days(name, np.nanvar, n)
    
    def nmaxvar_val_days(self, name, n=4):
        """Returns n max value days reduced with np.var along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmax_reduce_days(name, np.nanvar, n)
    
    def nminmean_val_days(self, name, n=4):
        """Returns n min value days reduced with np.mean along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmin_reduce_days(name, np.nanmean, n)
    
    def nmaxmean_val_days(self, name, n=4):
        """Returns n max value days reduced with np.mean along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmax_reduce_days(name, np.nanmean, n)
    
    def nminmed_val_days(self, name, n=4):
        """Returns n min value days reduced with np.median along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmin_reduce_days(name, np.nanmedian, n)
    
    def nmaxmed_val_days(self, name, n=4):
        """Returns n max value days reduced with np.median along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmax_reduce_days(name, np.nanmedian, n)
    
    def nminsum_val_days(self, name, n=4):
        """Returns n min value days reduced with np.sum along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmin_reduce_days(name, np.nanmean, n)
    
    def nmaxsum_val_days(self, name, n=4):
        """Returns n max value days reduced with np.sum along longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self._nmax_reduce_days(name, np.nanmean, n)

