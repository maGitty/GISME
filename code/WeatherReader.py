#!/usr/bin/python3
from glob_vars import lon_col, lat_col, era5_path, nuts0_shape

import os
import re
import numpy as np
import xarray as xr
import shapefile as shp
from datetime import datetime, timedelta, time
from shapely.geometry import Polygon, Point

class WeatherReader:
    """used to read nc files and to return xarray.DataArray containing desired data"""
    def __init__(self):
        """Set path to open files, store some information for faster response
        
        Returns
        -------
        None
        """
        assert os.path.exists(era5_path), 'path to weather data does not exist'
        
        self.filename = f'{era5_path}*.nc'
        with xr.open_mfdataset(self.filename) as nc_file:
            # drop times where no data is available, until now only seen at the end of the dataset
            self.wdata = nc_file #.dropna('time') # drops times with no values, carefully use, might throw away single points somewhere
        
        self.var_names = [name for name in self.wdata.data_vars]
        self.date_bounds = self.wdata['time'].min().values, self.wdata['time'].max().values
    
    def __nminmax_reduce_days(self, name, func, minmax, n):
        """Private method, return list of n days with min/max values for variable after
           applying function to reduce along longitude and latitude axes
        
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
        list of n days with min/max values for specified variable after applying func
        """
        assert (name in self.var_names and minmax in ['min', 'max']), f'wrong variable name ({name}) or minmax not in ["min","max"]'
        
        data = self.reduce_lonlat(name, func).dropna('time')
        data = data.sortby(data)
        n_minmax = data[:n] if minmax is 'min' else data[-n:]
        
        return n_minmax
    
    def __nmin_reduce_days(self, name, func, n):
        """Private method, return list of n days with min values for variable after
           applying function to reduce along longitude and latitude axes
        """
        return self.__nminmax_reduce_days(name, func, 'min', n)
    
    def __nmax_reduce_days(self, name, func, n):
        """Private method, return list of n days with max values for variable after
           applying function to reduce along longitude and latitude axes
        """
        return self.__nminmax_reduce_days(name, func, 'max', n)
    
    def reduce_lonlat(self, name, func):
        """Private method, return data for specified variable after applying
           function to reduce along longitude and latitude axes
        """
        assert name in self.var_names, f'column {name} not found'
        
        return self.wdata[name].reduce(func, dim=[lon_col,lat_col])
    
    def get_size(self):
        """Returns shape of whole data"""
        return self.wdata.sizes
    
    def get_coords(self):
        """Returns coordinating dimensions and respective value lists"""
        return self.wdata.coords
    
    def get_vars(self):
        """Returns list of variable names held by data"""
        return self.var_names
    
    def get_date_bounds(self):
        """Returns tuple of upper/lower date boundaries"""
        return self.date_bounds
    
    def get_lon(self):
        """Returns list of longitude values"""
        return self.wdata[lon_col].values
    
    def get_lat(self):
        """Returns list of latitude values"""
        return self.wdata[lat_col].values
    
    def get_long_name(self,var):
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
    
    def get_minmax(self,name):
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
        return (np.floor(self.wdata[name].min().values),np.ceil(self.wdata[name].max().values))
    
    def print_vars_texfmt(self):
        """Prints all variables in format 'name & unit & min & max' to just insert to latex
        
        Returns
        -------
        None
        """
        pow_regex = re.compile('(\*\*)(\S+)')
        for var in self.var_names:
            variable = self.wdata[var].dropna('time')
            unit = pow_regex.sub("^{\g<2>}",variable.units).replace(' ','~')
            print(f'{variable.long_name} & ${unit}$ & '
                  f'{variable.values.min().round(2):.2f} & '
                  f'{variable.values.max().round(2):.2f} \\\\')
    
    def check_isinDE(self):
        """Used to check which points of dataset are within germany
        
        Returns
        -------
        2D numpy.ndarray containing booleans wether point lies within germany or not
        """
        lons = self.get_coords()[lon_col].values
        lats = self.get_coords()[lat_col].values
        
        eu_shape = shp.Reader(nuts0_shape)
        
        for record in eu_shape.shapeRecords():
            if 'DE' in record.record:
                de_shape = record
                break
        
        poly = Polygon(de_shape.shape.points)
        
        coords = np.empty((len(lats),len(lons)),np.dtype(Point))
    
        for y in range(len(lats)):
            for x in range(len(lons)):
                lo = lons[x]
                la = lats[y]
                coords[y,x] = Point(lo,la)
    
        contains = np.vectorize(lambda p: p.within(poly) or p.touches(poly))
    
        contained = contains(coords)
        np.save(f'{data_path}isin', contained)
        
        return contained
    
    def check_isinRegion(self):
        """TODO
        """
        pass # TODO
    
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
            long name of variable
        
        is supposed to return a tuple of:
        dataframe containing values of desired variable ready for plotting with imshow
        """
        assert name in self.var_names, f'column {name} not found'

        # filter data by name and time
        data = self.wdata[name].sel(time=datetime)
        # update DataArray attributes to add min and max values for complete time
        data.attrs.update({'vmin':np.floor(self.wdata[name].min().values),'vmax':np.ceil(self.wdata[name].max().values)})
        
        return data
        

    def val4postime(self, name, lon, lat, dtime):
        """Returns value for variable at geographic position at specific time
        
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
        """Returns values for variable at geographic latitude at daytime averaged over all days
        
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
        """Returns variance over time reduced along longitude
           and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        
        Returns
        -------
        array for variable for each day reduced over longitude and latitude
        """
        assert name in self.var_names, f'column {name} not found'
        
        return self.reduce_lonlat(name, np.var)
    
    def nmin_val_days(self, name, n=4):
        """Returns n min value days reduced with np.min along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmin_reduce_days(name, np.min, n)

    def nmax_val_days(self, name, n=4):
        """Returns n max value days reduced with np.min along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmax_reduce_days(name, np.min, n)
    
    def nminvar_val_days(self, name, n=4):
        """Returns n min value days reduced with np.var along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmin_reduce_days(name, np.var, n)
    
    def nmaxvar_val_days(self, name, n=4):
        """Returns n max value days reduced with np.var along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmax_reduce_days(name, np.var, n)
    
    def nminmean_val_days(self, name, n=4):
        """Returns n min value days reduced with np.mean along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmin_reduce_days(name, np.mean, n)
    
    def nmaxmean_val_days(self, name, n=4):
        """Returns n max value days reduced with np.mean along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmax_reduce_days(name, np.mean, n)
    
    def nminmed_val_days(self, name, n=4):
        """Returns n min value days reduced with np.median along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmin_reduce_days(name, np.median, n)
    
    def nmaxmed_val_days(self, name, n=4):
        """Returns n max value days reduced with np.median along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmax_reduce_days(name, np.median, n)
    
    def nminsum_val_days(self, name, n=4):
        """Returns n min value days reduced with np.sum along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmin_reduce_days(name, np.mean, n)
    
    def nmaxsum_val_days(self, name, n=4):
        """Returns n max value days reduced with np.sum along
           longitude and latitude dimensions and drops NA values
        
        Parameters
        ----------
        name : string
               name of the variable
        n    : integer
               specifies number of days to return
        """
        return self.__nmax_reduce_days(name, np.mean, n)
    

rd = WeatherReader()
#print(rd.nmaxvar_val_days('t2m',50)['time'].values)
#rd.print_vars_texfmt()
#print(rd.get_date_bounds())
#rd.vals4time('t2m',datetime(2018,4,1,6))
#print(rd.val4postime('t2m', 10, 50, datetime(2017,1,1,12)))
#print(rd.vals4lat_daytime('t2m', 50, time(12)))
#print(rd.vals4time('t2m', datetime(2017,1,1,12)))
#lons = rd.get_coords()[lon].values
#lats = rd.get_coords()[lat].values

