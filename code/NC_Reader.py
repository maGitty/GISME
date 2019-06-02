#!/usr/bin/python3
from glob_vars import lon, lat

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta, time

class NC_Reader:
    """
    used to read nc files and to return pandas DataFrame containing desired data
    """
    def __init__(self, path):
        self.filename = path
        with xr.open_dataset(path) as nc_file:
            self.var_names = [name for name in nc_file.variables]
        
        self.lat = 'latitude'
        self.lon = 'longitude'
        
    def var4time(self, name, datetime):
        """Returns a tuple of: data, boundingbox to set plot range, long name of variable
        Parameters
        ----------
        name     : string
                   name of variable in nc file that should be contained in df
        datetime : datetime.datetime
                   index of time as number, 0=0, 1=6, 2=12, 3=18 o'clock
        
        is supposed to return a tuple of:
        dataframe containing values of desired variable ready for plotting with imshow
        bounding box to set the plot range
        long name to display as title or similar
        """
        if name not in self.var_names:
            print('Error: name not found, variable not in dataset')
            return
        with xr.open_dataset(self.filename) as nc_file:
            data = nc_file[name].sel(time=datetime).values
            lons = nc_file[lon].values
            lats = nc_file[lat].values
            bbox = (lons.min(), lons.max(), lats.min(), lats.max())
            long_name = f'{nc_file[name].long_name} ({nc_file[name].units})'
            #svar = nc_file.variables[name]
        
            ## zip size of each dimension with respective variable name to be able to index
            ##dims = [dim for dim in zip(nc_file.variables[name].dimensions, nc_file.variables[name].shape)]
            
            #lats = nc_file.variables[self.lat][:]
            #lons = nc_file.variables[self.lon][:]
            
            ## apply to convert from Kelvin to °C
            #df = pd.DataFrame(svar[:][time_index], index=lats, columns=lons)#.apply(lambda x: x-273.15)
            #bbox = (lons.min(), lons.max(), lats.min(), lats.max())
            #long_name = f'{svar.long_name} ({svar.units})'

        return (data, bbox, long_name)
        

    def var4postime(self, name, lon, lat, dtime):
        """Get values for variable at geographic position at specific time
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
        """
        if name not in self.var_names:
            print('Error: name not found, variable not in dataset')
            return
        
        with xr.open_dataset(self.filename) as nc_file:
            try:
                data = nc_file[name].sel(longitude=lon, latitude=lat, time=dtime).values
            except:
                print('Error: coordinates not within bbox')
                return None
        return data
    
    def var4lattime(self, name, lat, daytime):
        """Get values for variable at geographic position at daytime averaged over all days
        Parameters
        ----------
        name    : string
                  name of the variable
        lat     : numeric
                  the geographic latitude
        daytime : datetime.datetime
                  the specified daytime for which the data is returned
        """
        if name not in self.var_names:
            print('Error: name not found, variable not in dataset')
            return
        
        with xr.open_dataset(self.filename) as nc_file:
            try:
                # get fields for variable at latitude and daytime averaged over all days
                data = nc_file[name].sel(latitude=lat, time=daytime).values.mean(axis=0)
            except:
                print('Error: coordinates not within bbox')
                return None
        return data    
