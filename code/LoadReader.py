#!/usr/bin/python3

from glob_vars import load_path, cest_col,utc_col, de_load, hasMWlbl

import os
import numpy as np
import pandas as pd
import xarray as xr
from collections import OrderedDict
from datetime import datetime


class LoadReader:
    def __init__(self):
        """Initializes instance
        
        Returns
        -------
        None"""
        assert os.path.exists(load_path), 'file containing load data does not exist'
        
        with xr.open_dataset(load_path) as load_file:
            self.ldata = load_file
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
        return self.ldata.sizes
    
    def get_coords(self):
        """Returns coordinating dimensions and respective value lists"""
        return self.ldata.coords
    
    def get_vars(self):
        """Returns list of variable names held by data"""
        return self.var_names
    
    def get_date_bounds(self):
        """Returns tuple of upper/lower date boundaries"""
        return self.date_bounds
    
    def vals4time(self, name, time):
        """Returns values for variable for specified time"""
        assert name in self.var_names, f'column {name} not found'
        
        return self.ldata[name].sel(utc_timestamp=time)
    
    def vals4slice(self, name, start, stop, step=None):
        """Returns values for variable in specified time range"""
        assert name in self.var_names, f'column {name} not found'
        
        if step is None:
            # if no step specified simply return all values between start and stop
            return self.ldata[name].sel(utc_timestamp=slice(start,stop))
        else:
            # if step is given, return only desired points by passing a timeseries with frequency
            return self.ldata[name].sel(utc_timestamp=pd.date_range(start,stop,freq=f"{step}H"))
        
    
    def vals4times(self, name, timespec):
        """"""
        assert name in self.var_names, f'column {name} not found'
        
        return self.ldata[name].sel(utc_timestamp=timespec)
    
    
rd = LoadReader()
#rd._csv_to_nc()
#print(rd.val4time(de_load,datetime(2017,12,1,12)))
#print(rd.get_coords())
#print(rd.get_date_bounds())
#print(rd.get_size())
#print(rd.get_vars())

#class LoadReader:
    #def __init__(self, filetype='csv'):
        #self.load_path = load_path
        #self.filetype = filetype
        #if self.filetype is 'pkl':
            #self.load_path = self.load_path.replace('.csv', '.pkl')
        
    #def _to_netcdf(self):
        #"""used to convert csv file to .nc file format for speedup and compatibility"""
        #assert self.filetype is not 'nc', 'already has nc format'
        
        #load_df = self.__load_df()
        #ind = load_df[utc_col].values
        #ds = load_df.to_xarray()
        #dvars = [x for x in ds.data_vars]
        
        #"""
        #magic, set index for all vars to utc, then reset indexes which turns all variables to coordinates
        #then set utc to be a coordinate and reset all coordinates except utc to be variables again
        #in the end drop unused 'index' and cest time columns
        #"""
        #ds = ds.set_index({utc_col:dvars})
        #ds = ds.reset_index(dvars[1:])
        #ds = ds.set_coords(utc_col)
        #ds = ds.reset_coords(dvars[1:])
        #ds = ds.reset_index(['index',cest_col], drop=True)
        ## write to file with .nc type
        #ds.to_netcdf(self.load_path.replace(f'.{self.filetype}','.nc'))
        #self.filetype = 'nc'
        
    
    #def _to_pickle(self):
        #"""
        #used to convert csv file to pickle file format for faster read access
        #"""
        #assert self.filetype is not 'pkl', 'already has pkl format'
        
        #load_df = self.__load_df()
        
        #load_df[cest_col] = pd.to_datetime(load_df[cest_col], format='%Y-%m-%dT%H:%M:%S%z').to_timestamp()
        
        #temp_path = self.load_path.replace('.csv', f'.pkl')
        
        ## write to pickle file
        #load_df.to_pickle(temp_path)
        ## change filetype of reader
        #self.load_path = temp_path
        #self.filetype = 'pkl'
    
    #def __load_df(self):
        #if self.filetype is 'csv':
            #load_df = pd.read_csv(self.load_path, low_memory=False)
            #load_df[utc_col] = pd.to_datetime(load_df[utc_col])
            #return load_df
        #elif self.filetype is 'pkl':
            #return pd.read_pickle(self.load_path)
        #elif self.filetype is 'nc':
            #return xr.open_dataset(self.load_path)
        #else:
            #print('Error: wrong filetype')
            #return None
        
    #def __interpolate_missing(self):
        #"""
        #warning! might not work as expected, there might be unexpected behaviour, maybe is also filling values when 
        #"""
        #load_df = self.__load_df()
        #dr = pd.date_range(start=load_df[utc_col].min(), end=load_df[utc_col].max(), freq='15MIN', ambiguous='infer').tz_convert('Europe/Berlin')
        #print(load_df[cest_col][0], dr[0])
        #for i in range(len(load_df)):
            #if str(load_df[cest_col][i]) != str(dr[i]):
                #print(f'missing in line {i+1}', str(load_df[cest_col][i]), str(dr[i]))
        #load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=True).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        #load_df.set_index(load_df[cest_col])
        
        #drange = pd.date_range(start=load_df[cest_col].min(), end=load_df[cest_col].max(), freq='15MIN')
        #print(len(load_df), len(drange))
        #load_df = load_df.reindex(index=drange, fill_value=np.nan)
        #for col in load_df.columns:
            #load_df[col].interpolate(method='ffill', inplace=True)
        
        #load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=False).dt.tz_localize('Europe/Berlin', ambiguous='infer')
        #print(len(load_df))
    
    #def from_range(self, start, stop, step=False):
        #load_df = self.__load_df()
        ##load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=True).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        #if not step:
            #load_df = load_df[(load_df[cest_col] >= start) & (load_df[cest_col] <= stop)]
        ##load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=False).dt.tz_localize('Europe/Berlin', ambiguous='infer')
        #else:
            #load_df['hour'] = load_df[utc_col].dt.hour
            #load_df['minute'] = load_df[utc_col].dt.minute
            #load_df = load_df[load_df['hour'].isin([0,6,12,18]) & load_df['minute'].isin([0])]
            #load_df = load_df[(load_df[utc_col] >= start) & (load_df[utc_col] <= stop)]
            ##load_df = load_df[(load_df[cest_col] >= start) & (load_df[cest_col] <= stop)]
        #return load_df
    
    #def at_time(self, time):
        #load_df = self.__load_df()
        #return load_df[load_df[cest_col] == time]
    
    #def at_daytime(self, daytime):
        #load_df = self.__load_df()
        #load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=True).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)        
        #load_df = load_df.set_index(cest_col).at_time(daytime).reset_index()
        #load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=False).dt.tz_localize('Europe/Berlin', ambiguous='infer')        
        #return load_df






#def execute_example_range():
    #reader = LoadReader('pkl')
    ##reader.interpolate_missing()
    #start = pd.Timestamp(datetime(2017,1,1,12), tz='Europe/Berlin')
    #stop = pd.Timestamp(datetime(2017,1,1,13), tz='Europe/Berlin')
    #rng = reader.from_range(start, stop)
    #print(rng[cest_col].iloc[0])

#def execute_example_range1():
    #reader = LoadReader('pkl')
    ##reader.interpolate_missing()
    #start = pd.Timestamp(datetime(2017,1,1,0),tz='utc')
    #stop = pd.Timestamp(datetime(2017,1,6,18),tz='utc')
    #rng = reader.from_range(start, stop, step=True)
    #print(rng['DE_load_actual_entsoe_transparency'].get_values())

#def execute_example_daytime():
    #reader = LoadReader('pkl')
    #dt = reader.at_daytime('12:00')
    #print(dt.head())      

#def execute_example_interpolate():
    #reader = LoadReader('pkl')
    #reader.__interpolate_missing()
    ##start = pd.Timestamp(datetime(2017,1,1,12), tz='Europe/Berlin')
    ##stop = pd.Timestamp(datetime(2017,1,1,13), tz='Europe/Berlin')
    ##rng = reader.from_range(start, stop)
    ##print(rng[['utc_timestamp', cest_col]])    

##reader = LoadReader()
##reader._to_netcdf()

##execute_example_range1()
