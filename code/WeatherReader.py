#!/usr/bin/python3
from glob_vars import lon_col, lat_col, era5_path

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta, time

class WeatherReader:
    """
    used to read nc files and to return pandas DataFrame containing desired data
    """
    def __init__(self):
        """Set path to open files, store some information for faster response
        """
        assert os.path.exists(era5_path), 'path to weather data does not exist'
        
        self.filename = f'{era5_path}*.nc'
        with xr.open_mfdataset(self.filename) as nc_file:
            # drop times where no data is available, until now only seen at the end of the dataset
            self.wdata = nc_file.dropna('time')
        
        self.var_names = [name for name in self.wdata.data_vars]
        self.date_bounds = self.wdata['time'].min().values, self.wdata['time'].max().values
        #print(nc_file.dropna('time'))
    
    def __func_over_time(self, name, func):
        """
        Private method, return data for specified variable after applying
        function to reduce along longitude and latitude axes
        """
        assert name in self.var_names, f'column {name} not found'
        
        long_name = f'{self.wdata[name].long_name} ({self.wdata[name].units})'
        return self.wdata[name].reduce(func, dim=[lon_col,lat_col]), long_name
    
    def __nminmax_reduce_days(self, name, func, minmax, n):
        """
        Private method, return list of n days with min/max values for variable after
        applying function to reduce along longitude and latitude axes
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : int
               number of min/max values
              
        Returns
        -------
        list of n days with min/max values for specified variable after applying func
        """
        assert (name in self.var_names and minmax in ['min', 'max']), f'wrong variable name ({name}) or minmax not in ["min","max"]'
        
        data, long_name = self.__func_over_time(name, func)
        
        data = data.sortby(data)
        
        n_minmax = data[:n] if minmax is 'min' else data[-n:]
        
        return n_minmax, long_name
    
    def __nmin_reduce_days(self, name, func, n):
        """
        Private method, return list of n days with min values for variable after
        applying function to reduce along longitude and latitude axes
        """
        return self.__nminmax_reduce_days(name, func, 'min', n)
    
    def __nmax_reduce_days(self, name, func, n):
        """
        Private method, return list of n days with max values for variable after
        applying function to reduce along longitude and latitude axes
        """
        return self.__nminmax_reduce_days(name, func, 'max', n)
        
    def __merge_files(self, fname_list, out_path):
        """
        currently unused, use xr.open_mfdataset instead of merging, seems not to make a difference
        """
        if not fname_list:
            print('Error: empty list!')
            return
        big_nc = xr.open_dataset(fname_list.pop(), decode_times=True)
        
        while True:
            if not fname_list:
                print('list is empty')
                return
            small_nc = xr.open_dataset(fname_list.pop(), decode_times=True)
            big_nc = xr.merge([big_nc, small_nc])
            small_nc.close()
            
        big_nc.to_netcdf(path=out_path, mode='w')
        big_nc.close()
    
    def get_size(self):
        """Returns shape of whole data"""
        return self.wdata.sizes
    
    def get_coords(self):
        """Returns coordinating dimensions and respective value lists"""
        return self.wdata.coords
    
    def get_vars(self):
        """returns list of variable names held by data"""
        return self.var_names
    
    def get_date_bounds(self):
        """returns tuple of upper/lower date boundaries"""
        return self.date_bounds
    
    def get_lon(self):
        """returns list of longitude values"""
        return self.wdata[lon_col].values
    
    def get_lat(self):
        """returns list of latitude values"""
        return self.wdata[lat_col].values
        
    def vals4time(self, name, datetime):
        """Returns the values for specified variable and time
        
        Parameters
        ----------
        name     : string
                   name of variable in nc file that should be contained in df
        datetime : datetime.datetime
                   the specified datetime for which the data is returned
        
        Returns
        -------
        tuple of :
            2D data with values for variable over latitude and longitude
            boundingbox quadruple to set plot range
            long name of variable
        
        is supposed to return a tuple of:
        dataframe containing values of desired variable ready for plotting with imshow
        bounding box to set the plot range
        long name to display as title or similar
        """
        assert name in self.var_names, f'column {name} not found'

        minmax = np.floor(self.wdata[name].min()), np.ceil(self.wdata[name].max())
        data = self.wdata[name].sel(time=datetime).values
        lons = self.wdata[lon_col].values
        lats = self.wdata[lat_col].values
        bbox = (lons.min(), lons.max(), lats.min(), lats.max())
        long_name = f'{self.wdata[name].long_name} ({self.wdata[name].units})'

        return data, bbox, long_name, minmax
        

    def val4postime(self, name, lon, lat, dtime):
        """Get value for variable at geographic position at specific time
        
        Parameters
        ----------
        name  : string
                name of the variable
        lon   : numeric
                the geographic longitude
        lat   : numeric
                the geographic latitude
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
            raise Exception(f'coordinates not within bbox? lon:{lon}, lat:{lat}')
        return data
    
    def vals4lon_daytime(self, name, longitude, daytime):
        """Get values for variable at geographic longitude at daytime averaged over all days
        
        Parameters
        ----------
        name      : string
                    name of the variable
        longitude : numeric
                    the geographic longitude
        daytime   : datetime.datetime
                    the specified daytime for which the data is returned
        
        Returns
        -------
        values for specified variable and longitude along latitude and daytime averaged over all days
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
        """Get values for variable at geographic latitude at daytime averaged over all days
        
        Parameters
        ----------
        name     : string
                   name of the variable
        latitude : numeric
                   the geographic latitude
        daytime  : datetime.datetime
                   the specified daytime for which the data is returned
        
        Returns
        -------
        values for specified variable and latitude along longitude and daytime averaged over all days
        """
        assert name in self.var_names, f'column {name} not found'

        try:
            long_name = f'{self.wdata[name].long_name} ({self.wdata[name].units})'
            # get fields for variable at latitude and daytime averaged over all days
            data = self.wdata[name].sel(latitude=latitude, time=daytime).mean(dim='time')
        except:
            raise Exception(f'coordinates not within bbox? lat:{latitude}')
        return data, long_name
    
    def var_over_time(self, name):
        """
        Returns variance over time reduced along longitude
        and latitude dimensions and drops NA values
        """
        assert name in self.var_names, f'column {name} not found'
        
        return self.__func_over_time(name, np.var)
    
    def nmin_val_days(self, name, n=4):
        """
        Returns n min value days reduced with np.min along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmin_reduce_days(name, np.min, n)

    def nmax_val_days(self, name, n=4):
        """
        Returns n max value days reduced with np.min along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmax_reduce_days(name, np.min, n)
    
    def nminvar_val_days(self, name, n=4):
        """
        Returns n min value days reduced with np.var along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmin_reduce_days(name, np.var, n)
    
    def nmaxvar_val_days(self, name, n=4):
        """
        Returns n max value days reduced with np.var along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmax_reduce_days(name, np.var, n)
    
    def nminmean_val_days(self, name, n=4):
        """
        Returns n min value days reduced with np.mean along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmin_reduce_days(name, np.mean, n)
    
    def nmaxmean_val_days(self, name, n=4):
        """
        Returns n max value days reduced with np.mean along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmax_reduce_days(name, np.mean, n)
    
    def nminmed_val_days(self, name, n=4):
        """
        Returns n min value days reduced with np.median along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmin_reduce_days(name, np.median, n)
    
    def nmaxmed_val_days(self, name, n=4):
        """
        Returns n max value days reduced with np.median along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmax_reduce_days(name, np.median, n)
    
    def nminsum_val_days(self, name, n=4):
        """
        Returns n min value days reduced with np.sum along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmin_reduce_days(name, np.mean, n)
    
    def nmaxsum_val_days(self, name, n=4):
        """
        Returns n max value days reduced with np.sum along
        longitude and latitude dimensions and drops NA values
        """
        return self.__nmax_reduce_days(name, np.mean, n)
    

rd = WeatherReader()
#print(rd.get_date_bounds())
#print(rd.vals4time('t2m',datetime(2018,4,1,6)))
#print(rd.val4postime('t2m', 10, 50, datetime(2017,1,1,12)))
#print(rd.vals4lat_daytime('t2m', 50, time(12)))
#print(rd.vals4time('t2m', datetime(2017,1,1,12)))
#lons = rd.get_coords()[lon].values
#lats = rd.get_coords()[lat].values
#print([[x,y] for x in lons for y in lats])
#print(rd.nmin_val_days('t2m')[0]['time'].values)
#print(rd.var_over_time('t2m'))
