#!/usr/bin/python3

from glob_vars import load_path, load_file_time_cols, PKL, cest_col,utc_col

import numpy as np
import pandas as pd
from datetime import datetime

class LoadReader:
    def __init__(self, filetype='csv'):
        self.load_path = load_path
        self.filetype = filetype
        if self.filetype is 'pkl':
            self.load_path = self.load_path.replace('.csv', f'.{PKL}')
        
    def _to_pickle(self):
        """
        used to convert csv file to pickle file format for faster read access
        """
        if self.filetype is 'pkl':
            print('reader is already pickled')
            return
        load_df = self._load_df()
        
        load_df[cest_col] = pd.to_datetime(load_df[cest_col], format='%Y-%m-%dT%H:%M:%S%z').to_timestamp()
        
        temp_path = self.load_path.replace('.csv', f'.{PKL}')
        
        # write to pickle file
        load_df.to_pickle(temp_path)
        # change filetype of reader
        self.load_path = temp_path
        self.filetype = PKL
    
    def _load_df(self):
        if self.filetype is 'csv':
            load_df = pd.read_csv(self.load_path, low_memory=False)
            load_df[cest_col] = pd.to_datetime(load_df[cest_col], format='%Y-%m-%dT%H:%M:%S%z').to_timestamp()
            return load_df
        elif self.filetype is PKL:
            return pd.read_pickle(self.load_path)
        else:
            print('Error: wrong filetype')
            return None
        
    def _interpolate_missing(self):
        """
        warning! might not work as expected, there might be unexpected behaviour, maybe is also filling values when 
        """
        load_df = self._load_df()
        dr = pd.date_range(start=load_df[utc_col].min(), end=load_df[utc_col].max(), freq='15MIN', ambiguous='infer').tz_convert('Europe/Berlin')
        print(load_df[cest_col][0], dr[0])
        for i in range(len(load_df)):
            if str(load_df[cest_col][i]) != str(dr[i]):
                print(f'missing in line {i+1}', str(load_df[cest_col][i]), str(dr[i]))
        load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=True).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        load_df.set_index(load_df[cest_col])
        
        drange = pd.date_range(start=load_df[cest_col].min(), end=load_df[cest_col].max(), freq='15MIN')
        print(len(load_df), len(drange))
        load_df = load_df.reindex(index=drange, fill_value=np.nan)
        for col in load_df.columns:
            load_df[col].interpolate(method='ffill', inplace=True)
        
        load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=False).dt.tz_localize('Europe/Berlin', ambiguous='infer')
        print(len(load_df))
    
    def from_range(self, start, stop):
        load_df = self._load_df()
        #load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=True).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)        
        load_df = load_df[(load_df[cest_col] >= start) & (load_df[cest_col] <= stop)]
        #load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=False).dt.tz_localize('Europe/Berlin', ambiguous='infer')
        return load_df
    
    def at_time(self, time):
        load_df = self._load_df()
        return load_df[load_df[cest_col] == time]
    
    def at_daytime(self, daytime):
        load_df = self._load_df()
        load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=True).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)        
        load_df = load_df.set_index(cest_col).at_time(daytime).reset_index()
        load_df[cest_col] = pd.to_datetime(load_df[cest_col], utc=False).dt.tz_localize('Europe/Berlin', ambiguous='infer')        
        return load_df






def execute_example_range():
    reader = LoadReader('pkl')
    #reader.interpolate_missing()
    start = pd.Timestamp(datetime(2017,1,1,12), tz='Europe/Berlin')
    stop = pd.Timestamp(datetime(2017,1,1,13), tz='Europe/Berlin')
    rng = reader.from_range(start, stop)
    print(rng[['utc_timestamp', cest_col]])

def execute_example_daytime():
    reader = LoadReader('pkl')
    dt = reader.at_daytime('12:00')
    print(dt.head())      

def execute_example_interpolate():
    reader = LoadReader('pkl')
    reader._interpolate_missing()
    #start = pd.Timestamp(datetime(2017,1,1,12), tz='Europe/Berlin')
    #stop = pd.Timestamp(datetime(2017,1,1,13), tz='Europe/Berlin')
    #rng = reader.from_range(start, stop)
    #print(rng[['utc_timestamp', cest_col]])    


execute_example_interpolate()