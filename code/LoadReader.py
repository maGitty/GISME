#!/usr/bin/python3

from glob_vars import load_path, cest_col,utc_col, de_load, hasMWlbl

import os
import numpy as np
import pandas as pd
import xarray as xr
from collections import OrderedDict
from datetime import datetime


class LoadReader:
    """Used to read load data and to return data respectively"""
    def __init__(self):
        """Initializes instance
        
        Returns
        -------
        None
        """
        assert os.path.exists(load_path), 'file containing load data does not exist'
        
        with xr.open_dataset(load_path) as load_file:
            self.__ldata__ = load_file
            self.var_names = [name for name in load_file.data_vars]
            self.date_bounds = load_file[utc_col].min().values, load_file[utc_col].max().values
            #print(load_file['DE_load_actual_entsoe_transparency'].values)
    
    def _csv_to_nc(self):
        """Used to convert csv file to .nc file format for speedup and compatibility
        
        Returns
        -------
        None
        """
        load_file = pd.read_csv(f'{os.path.splitext(load_path)[0]}.csv',header=0,index_col=0)
        load_file.index = pd.to_datetime(load_file.index)
        
        # convert to xarray.Dataset and drop unused local time column
        ds = load_file.to_xarray().drop(cest_col)
        self.var_names = [var for var in ds.data_vars]
        
        # convert again to datetime, won't be recognized otherwise
        ds[utc_col] = pd.to_datetime(ds[utc_col])
        
        # set unit variable for each variable; all MW, exept for some shares
        for var in self.var_names:
            unit = 'MW' if any(label in var for label in hasMWlbl) else 'share'
            ds[var].attrs.update(OrderedDict([('units', unit)]))
        
        # write to file
        ds.to_netcdf(load_path)
    
    def get_size(self):
        """Returns shape of whole data"""
        return self.__ldata__.sizes
    
    def get_coords(self):
        """Returns coordinating dimensions and respective value lists"""
        return self.__ldata__.coords
    
    def get_vars(self):
        """Returns list of variable names held by data"""
        return self.var_names
    
    def get_date_bounds(self):
        """Returns tuple of upper/lower date boundaries"""
        return self.date_bounds
    
    def vals4time(self, name, time):
        """Returns values for variable for specified time/s
        
        Parameters
        ----------
        name   : string
                 name of the variable
        time   : pandas.Timestamp, string, slice or DatetimeIndex (all arguments taken by xarrays 'sel')
                 time for which value is to be returned
        
        Returns
        -------
        xarray.DataArray containing desired data with respective timestamps"""
        assert name in self.var_names, f'column {name} not found'
        
        return self.__ldata__[name].sel(utc_timestamp=time).interpolate_na('utc_timestamp')
        
    def vals4slice(self, name, start, stop, step=None):
        """Returns values for variable in specified time range
        
        Parameters
        ----------
        name   : string
                 name of the variable
        start  : pandas.Timestamp
                 start time for slice
        stop   : pandas.Timestamp
                 end time for slice
        step   : None or integer
                 step for timestamps between start end end time in hours,
        
        Returns
        -------
        xarray.DataArray containing desired data with respective timestamps
        """
        assert (name in self.var_names), f'column {name} not found'
        
        if step is None:
            # if no step specified simply return all values between start and stop
            return self.__ldata__[name].sel(utc_timestamp=slice(start,stop)).interpolate_na('utc_timestamp')
        else:
            # if step is given, return only desired points by passing a timeseries with frequency
            return self.__ldata__[name].sel(utc_timestamp=pd.date_range(start,stop,freq=f"{step}H")).interpolate_na('utc_timestamp')
    
    def vals4step(self,name,step=1):
        """Returns complete values for specified step size
        
        Parameters
        ----------
        name   : string
                 name of the variable
        step   : None or integer
                 step for timestamps between start end end time in hours,
        
        Returns
        -------
        xarray.DataArray containing desired data with respective timestamps
        """
        return self.__ldata__[name].sel(utc_timestamp=pd.date_range(*self.date_bounds,freq=f"{step}H")).interpolate_na('utc_timestamp')
        
    
    
# rd = LoadReader()

#start = datetime(2015,1,1)
#stop = datetime(2017,12,31)

#data = rd.vals4slice(de_load, start, stop, step=2).values
#print(data)
