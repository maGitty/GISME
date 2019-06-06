#!/usr/bin/python3
from glob_vars import lon, lat, era5_path

from netCDF4 import Dataset
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta, time

class NC_Reader:
    """
    used to read nc files and to return pandas DataFrame containing desired data
    """
    def __init__(self):
        """
        raise AssertionError if path doesn't exist
        """
        assert(os.path.exists(era5_path))
        
        self.filename = f'{era5_path}*.nc'
        with xr.open_mfdataset(f'{era5_path}*.nc') as nc_file:
            self.var_names = [name for name in nc_file.data_vars]
            self.coords = nc_file.coords
            self.size = nc_file.sizes
    
    def get_size(self):
        return self.size
    
    def get_coords(self):
        return self.coords
    
    def get_vars(self):
        return self.var_names
        
    def merge_files(self, fname_list, out_path):
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
            
        big_nc.to_netcdf(path=out_path, mode='w')
        
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
        assert(name in self.var_names)

        with xr.open_mfdataset(self.filename) as nc_file:
            minmax = np.floor(nc_file[name].min()), np.ceil(nc_file[name].max())
            data = nc_file[name].sel(time=datetime).values
            lons = nc_file[lon].values
            lats = nc_file[lat].values
            bbox = (lons.min(), lons.max(), lats.min(), lats.max())
            long_name = f'{nc_file[name].long_name} ({nc_file[name].units})'

            return (data, bbox, long_name, minmax)
        

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
        assert(name in self.var_names)
        
        with xr.open_mfdataset(self.filename) as nc_file:
            try:
                data = nc_file[name].sel(longitude=lon, latitude=lat, time=dtime).values
            except:
                print('Error: coordinates not within bbox')
                return None
        return data
    
    def vals4lattime(self, name, lat, daytime):
        """Get values for variable at geographic position at daytime averaged over all days
        
        Parameters
        ----------
        name    : string
                  name of the variable
        lat     : numeric
                  the geographic latitude
        daytime : datetime.datetime
                  the specified daytime for which the data is returned
        
        Returns
        -------
        values for specified variable and latitude along longitude and daytime averaged over all days
        """
        assert(name in self.var_names)

        with xr.open_mfdataset(self.filename) as nc_file:
            try:
                # get fields for variable at latitude and daytime averaged over all days
                data = nc_file[name].sel(latitude=lat, time=daytime).values.mean(axis=0)
            except:
                print('Error: coordinates not within bbox')
                return None
        return data
    
    def _nminmax_reduce_days(self, name, func, minmax, n=4):
        """Return list of n days with min/max values for var after applying func
        
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
        assert(name in self.var_names and minmax in ['min', 'max'])
        
        with xr.open_mfdataset(self.filename) as nc_file:
            nc_file = nc_file[name].reduce(func, dim=[lon,lat]).dropna(dim='time')
            nc_file = nc_file.sortby(nc_file)
            
            n_minmax = nc_file[:n] if minmax is 'min' else nc_file[-n:]
        
        return n_minmax
    
    def _nmin_reduce_days(self, name, func, n=4):
        return self._nmin_reduce_days(name, func, n)
    
    def _nmax_reduce_days(self, name, func, n=4):
        return self._nmax_reduce_days(name, func, n)
    
    def nmin_val_days(self, name, n=4):
        return self._nmin_reduce_days(name, np.min, n)

    def nmax_val_days(self, name, n=4):
        return self._nmax_reduce_days(name, np.min, n)
    
    def nminvar_val_days(self, name, n=4):
        return self._nmin_reduce_days(name, np.var, n)
    
    def nmaxvar_val_days(self, name, n=4):
        return self._nmax_reduce_days(name, np.var, n)
    
    def nminmean_val_days(self, name, n=4):
        return self._nmin_reduce_days(name, np.mean, n)
    
    def nmaxmean_val_days(self, name, n=4):
        return self._nmax_reduce_days(name, np.mean, n)
    
    def nminmed_val_days(self, name, n=4):
        return self._nmin_reduce_days(name, np.median, n)
    
    def nmaxmed_val_days(self, name, n=4):
        return self._nmax_reduce_days(name, np.median, n)
    
    def nminsum_val_days(self, name, n=4):
        return self._nmin_reduce_days(name, np.mean, n)
    
    def nmaxsum_val_days(self, name, n=4):
        return self._nmax_reduce_days(name, np.mean, n)
    
    def _func_over_time(self, name, func):
        assert(name in self.var_names)
        
        with xr.open_mfdataset(self.filename) as nc_file:
            return nc_file[name].reduce(func, dim=[lon,lat]).dropna(dim='time')
    
    def var_over_time(self, name):
        assert(name in self.var_names)

        with xr.open_mfdataset(self.filename) as nc_file:
            return nc_file[name].var(dim=[lon,lat])

#rd = NC_Reader()
#lons = rd.get_coords()[lon].values
#lats = rd.get_coords()[lat].values
#print([[x,y] for x in lons for y in lats])
#print(rd.nmin_val_days('t2m')[0]['time'].values)
