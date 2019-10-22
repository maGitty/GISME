#!/usr/bin/python3

"""
In this module, the different methods for forecasting are implemented
"""

from gisme import (de_load, data_path, log)
from gisme.LoadReader import LoadReader
from gisme.WeatherReader import WeatherReader

from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.regression.linear_model import OLS
import pandas as pd
import numpy as np
import pickle
import os

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class TSForecast:
    """Used to store time series forecast"""
    def __init__(self, hours_ahead, forecast, actual, train_start, train_stop, fc_start, fc_stop):
        """Initializes instance
        
        Parameters
        ----------
        hours_ahead : int
                      specifies how many hours ahead forecast is done
        forecast    : numpy.ndarray
                      contains forecast data
        actual      : numpy.ndarray
                      contains actual data
        train_start : datetime.datetime
                      training start time
        train_stop  : datetime.datetime
                      training stop time
        fc_start    : datetime.datetime
                      forecast start time
        fc_stop     : datetime.datetime
                      forecast stop time
        
        Returns
        -------
        None
        """
        self.date_range = pd.date_range(fc_start, fc_stop, freq='1H')
        assert (forecast.size == actual.size) and (forecast.size == self.date_range.size),\
            f'lengths do not equal: fc:{forecast.size},load:{actual.size},hours:{self.date_range.size}'
        self.hours_ahead = hours_ahead
        self.forecast = forecast
        self.actual = actual
        self.train_start = train_start
        self.train_stop = train_stop
        self.fc_start = fc_start
        self.fc_stop = fc_stop

    def __str__(self):
        """Returns a summary of this forecast
        
        Returns
        -------
        A summary as string
        """
        sum_str = f"{self.hours_ahead} hours ahead forecast from {self.fc_start} "\
            f"to {self.fc_stop} trained from {self.train_start} to {self.train_stop}:"\
            f"\nrmse : {np.round(self.rmse(), 3)} , mae : {np.round(self.mae(), 3)} ,"\
            f" mpe : {np.round(self.mpe(), 3)} , mape : {np.round(self.mape(), 3)}"
        return sum_str
    
    def __teXstr__(self):
        """Returns the error measures of this forecast
        
        Returns
        -------
        The error measures as string in tex format to insert into a table only missing the type of the model
        """
        return f"{np.round(self.rmse(), 3)} & {np.round(self.mae(), 3)} & "\
               f"{np.round(self.mpe(), 3)} & {np.round(self.mape(), 3)}\\\\"
    
    def __plot__(self):
        """TODO
        """
        date_range = pd.date_range(self.fc_start, self.fc_stop, freq='1H')
        plt.plot(date_range, self.actual, label='actual')
        plt.plot(date_range, self.forecast, label=f'{self.hours_ahead}H forecast')
        plt.legend()
    
    def sel(self, start, stop):
        """TODO
        """
        df = pd.DataFrame(data={'actual': self.actual, 'forecast': self.forecast},
                          index=self.date_range)
        return df[start:stop]
    
    def difference(self, other_fc):
        """Return difference between values of
           this forecast and other_fc
        
        Parameters
        ----------
        other_fc : TSForecast
                   forecast to compare to this forecast
        
        Returns
        -------
        numpy.array of differences between forecasts
        """
        assert self.actual.size == other_fc.actual.size,\
            f'forecast lengths are different: {len(self.actual)}, {len(other_fc.actual)}'
        return self.forecast-other_fc.forecast
    
    def mae(self):
        """Calculates Mean Absolute Error of this forecast
        
        Returns
        -------
        Mean Absolute Error as float
        """
        return np.mean(np.abs(self.actual-self.forecast))
    
    def mpe(self):
        """Calculates Mean Percentage Error of this forecast
        
        Returns
        -------
        Mean Percentage Error as float
        """
        return np.mean(((self.actual-self.forecast)/self.actual)*100)
    
    def mape(self):
        """Calculates Mean Absolute Percentage Error of this forecast
        
        Returns
        -------
        Mean Absolute Percentage Error as float
        """
        return np.mean(np.abs((self.actual-self.forecast)/self.actual)*100)
    
    def rmse(self):
        """Calculates Root Mean Square Error
        
        Returns
        -------
        Root Mean Square Error as float
        """
        return np.sqrt(((self.forecast-self.actual)**2).mean())
    
    def summary(self):
        """Returns a summary of this forecast
        
        Returns
        -------
        A summary as string
        """
        sum_str = f"{self.hours_ahead} hours ahead forecast from {self.fc_start} "\
            f"to {self.fc_stop}\ntrained from {self.train_start} to {self.train_stop}:"\
            f"\nmape : {np.round(self.mape(),2)} , rmse : {np.round(self.rmse(),2)}"
        
        return sum_str


class ARMAXForecast:
    """ARMAX forecast"""
    def __init__(self, start, stop, p=None, q=None, exog=None, const=True):
        """Initializes instance
        
        Parameters
        ----------
        start : datetime.datetime
                start time of forecast
        stop  : datetime.datetime
                stop time of forecast
        p     : int
                parameter for autoregressive part
        q     : int
                parameter for moving average part
        exog  : list of strings
                used exogenous variables, see __load_data for possible choices
        const : boolean
                specifies whether to include a constant or not
        
        Returns
        -------
        None
        """
        self.load_reader = LoadReader()
        self.weather_reader = WeatherReader()
        self.p = p
        self.q = q
        self.start = start
        self.stop = stop
        self.__armax_result = None
        self.forecasts = []
        self.exog = exog
        self.has_const = const
        self.__load_data()
        
    def __load_data(self):
        """Loads data including exogenous variables as pandas.DataFrame based on self.exog
           possible choices for exogenous variables:
             - 'load_lag'     : load data shifted by one week (with this parameter only data from 2015/01/08 can be used)
             - 'dayofweek'    : week day (one dummy per day)
             - 'weekend'      : dummy for whether it is weekend or not (1 or 0)
             - 'data_counter' : counting data points beginning from 0
             - 't2m_max'      : maximum of 2 metre temperature for each step
             - 't2m_mean'     : mean of 2 metre temperature for each step
             - 't2m_top10'    : top 10 regions grid points compared by population
             - 't2m_all'      : every grid point of 2 metre temperature as single exogenous variable
             - 'u10_top10'    : top 10 regions grid points compared by population
             - 'u10_all'      : every grid point of 10 metre U wind component as single exogenous variable
             - 'v10_top10'    : top 10 regions grid points compared by population
             - 'v10_all'      : every grid point of 10 metre V wind component as single exogenous variable
             - 'lai_hv_top10' : top 10 regions grid points compared by population
             - 'lai_hv_all'   : every grid point of high vegetation leaf area index as single exogenous variable
             - 'lai_lv_top10' : top 10 regions grid points compared by population
             - 'lai_lv_all'   : every grid point of low vegetation leaf area index as single exogenous variable
             - 'lcc_top10'    : top 10 regions grid points compared by population
             - 'lcc_all'      : every grid point of low cloud cover as single exogenous variable
             - 'stl1_top10'   : top 10 regions grid points compared by population
             - 'stl1_all'     : every grid point of soil temperature level 1 as single exogenous variable
             - 'slhf_top10'   : top 10 regions grid points compared by population
             - 'slhf_all'     : every grid point of surface latent heat flux as single exogenous variable
             - 'str_top10'    : top 10 regions grid points compared by population
             - 'str_all'      : every grid point of surface net thermal radiation as single exogenous variable
             - 'sshf_top10'   : top 10 regions grid points compared by population
             - 'sshf_all'     : every grid point of surface sensible heat flux as single exogenous variable
             - 'tcc_top10'    : top 10 regions grid points compared by population
             - 'tcc_all'      : every grid point of total cloud cover as single exogenous variable
             - 'tcrw_top10'   : top 10 regions grid points compared by population
             - 'tcrw_all'     : every grid point of total column rain water as single exogenous variable
             - 'fdir_top10'   : top 10 regions grid points compared by population
             - 'fdir_all'     : every grid point of total sky direct solar radiation at surface as single exogenous variable
             """
        last_date = datetime(2018, 12, 31)
        
        load_data = self.load_reader.vals4slice(de_load, self.start, last_date, step=1)
        date_range = pd.date_range(self.start, last_date, freq='1H')
        self.load_data = pd.DataFrame(data={'load': load_data}, index=date_range)
        self.ex_data = pd.DataFrame(data=[], index=date_range)
        
        if self.exog is None:
            return
        
        dow = date_range.to_series().dt.dayofweek
        if 'load_lag' in self.exog:
            delta1week = timedelta(weeks=1)
            self.ex_data['load_shift'] = self.load_reader.vals4slice(de_load, self.start-delta1week,
                                                                     last_date-delta1week, step=1)
        if 'dayofweek' in self.exog:
            self.ex_data['Monday'] = (dow == 0).astype(int)
            self.ex_data['Tuesday'] = (dow == 1).astype(int)
            self.ex_data['Wednesday'] = (dow == 2).astype(int)
            self.ex_data['Thursday'] = (dow == 3).astype(int)
            self.ex_data['Friday'] = (dow == 4).astype(int)
            self.ex_data['Saturday'] = (dow == 5).astype(int)
            self.ex_data['Sunday'] = (dow == 6).astype(int)
        if 'weekend' in self.exog:
            self.ex_data['weekend'] = dow.isin([5, 6]).astype(int)
        if 'data_counter' in self.exog:
            self.ex_data['data_counter'] = range(len(load_data))
        if 't2m_max' in self.exog:
            self.ex_data['t2m_max'] = self.weather_reader.max_slice('t2m', self.start, last_date)
        if 't2m_mean' in self.exog:
            self.ex_data['t2m_mean'] = self.weather_reader.mean_slice('t2m', self.start, last_date)
        if 't2m_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('t2m', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f't2m_top{i}gridpoint'] = top_i
        if 't2m_all' in self.exog:
            for i, t2m in enumerate(self.weather_reader.flattened_slice('t2m', self.start, last_date)):
                self.ex_data[f't2m{i}'] = t2m
        if 'u10_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('u10', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'u10_top{i}gridpoint'] = top_i
        if 'u10_all' in self.exog:
            for i, u10 in enumerate(self.weather_reader.flattened_slice('u10', self.start, last_date)):
                self.ex_data[f'u10{i}'] = u10
        if 'v10_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('v10', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'v10_top{i}gridpoint'] = top_i
        if 'v10_all' in self.exog:
            for i, v10 in enumerate(self.weather_reader.flattened_slice('v10', self.start, last_date)):
                self.ex_data[f'v10{i}'] = v10
        if 'lai_hv_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('lai_hv' , self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'lai_hv_top{i}gridpoint'] = top_i
        if 'lai_hv_all' in self.exog:
            for i, lai_hv in enumerate(self.weather_reader.flattened_slice('lai_hv', self.start, last_date)):
                self.ex_data[f'lai_hv{i}'] = lai_hv
        if 'lai_lv_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('lai_lv', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'lai_lv_top{i}gridpoint'] = top_i
        if 'lai_lv_all' in self.exog:
            for i, lai_lv in enumerate(self.weather_reader.flattened_slice('lai_lv', self.start, last_date)):
                self.ex_data[f'lai_lv{i}'] = lai_lv
        if 'lcc_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('lcc', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'lcc_top{i}gridpoint'] = top_i
        if 'lcc_all' in self.exog:
            for i, lcc in enumerate(self.weather_reader.flattened_slice('lcc', self.start, last_date)):
                self.ex_data[f'lcc{i}'] = lcc
        if 'stl1_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('stl1', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'stl1_top{i}gridpoint'] = top_i
        if 'stl1_all' in self.exog:
            for i, stl1 in enumerate(self.weather_reader.flattened_slice('stl1', self.start, last_date)):
                self.ex_data[f'stl1{i}'] = stl1
        if 'slhf_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('slhf', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'slhf_top{i}gridpoint'] = top_i
        if 'slhf_all' in self.exog:
            for i, slhf in enumerate(self.weather_reader.flattened_slice('slhf', self.start, last_date)):
                self.ex_data[f'slhf{i}'] = slhf
        if 'str_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('str', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'str_top{i}gridpoint'] = top_i
        if 'str_all' in self.exog:
            for i, str_var in enumerate(self.weather_reader.flattened_slice('str', self.start, last_date)):
                self.ex_data[f'str{i}'] = str_var
        if 'sshf_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('sshf', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'sshf_top{i}gridpoint'] = top_i
        if 'sshf_all' in self.exog:
            for i, sshf in enumerate(self.weather_reader.flattened_slice('sshf', self.start, last_date)):
                self.ex_data[f'sshf{i}'] = sshf
        if 'tcc_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('tcc', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'tcc_top{i}gridpoint'] = top_i
        if 'tcc_all' in self.exog:
            for i, tcc in enumerate(self.weather_reader.flattened_slice('tcc', self.start, last_date)):
                self.ex_data[f'tcc{i}'] = tcc
        if 'tcrw_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('tcrw', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'tcrw_top{i}gridpoint'] = top_i
        if 'tcrw_all' in self.exog:
            for i, tcrw in enumerate(self.weather_reader.flattened_slice('tcrw', self.start, last_date)):
                self.ex_data[f'tcrw{i}'] = tcrw
        if 'fdir_top10' in self.exog:
            for i, top_i in enumerate(self.weather_reader.demo_top_n_regions_slice('fdir', self.start,
                                                                                   last_date, 10).transpose()):
                self.ex_data[f'fdir_top{i}gridpoint'] = top_i
        if 'fdir_all' in self.exog:
            for i, fdir in enumerate(self.weather_reader.flattened_slice('fdir', self.start, last_date)):
                self.ex_data[f'fdir{i}'] = fdir
    
    def __load_exog(self, tstart, tstop):
        """Loads exogenous variables as numpy.ndarray based on self.exog
        
        Parameters
        ----------
        tstart      : datetime.datetime
                      start time
        tstop       : datetime.datetime
                      stop time
        range_start : int
                      if existing, specifies start point of data_counter
              
        Returns
        -------
        numpy.ndarray of exogenous data
        """
        if self.exog is None:
            return
        return self.ex_data[tstart:tstop].values

    @staticmethod
    def load(fname):
        """Loads ARMAX_forecast instance from file name
        
        Parameters
        ----------
        fname : string
                path to pickled file containing instance
        
        Returns
        -------
        ARMAX_forecast instance if exists, raising Exception otherwise
        """
        return pickle.load(open(fname,'rb'))
    
    def save(self):
        """Stores instance to pickle
        
        Returns
        -------
        file name of stored instance as string
        """
        dir_pth = os.path.join(data_path,'ARMAX')
        if not os.path.exists(dir_pth):
            os.mkdir(dir_pth)
        fname = os.path.join(data_path,'ARMAX',
                             f'start{self.start.strftime("%Y%m%d%H")}_stop'\
                             f'{self.stop.strftime("%Y%m%d%H")}_p{self.p}q{self.q}'\
                             f'{"" if self.exog is None else "_"+"_".join(self.exog)}.pkl')
        log.info(f'saving armax as {fname}')
        pickle.dump(self,open(fname,'wb'))
        return fname
    
    def train(self):
        """Fits ARMAX_forecast instance with specified endogenous and exogenous data
           Needs to be called before forecasting
        
        Returns
        -------
        None
        """
        data = self.load_data[self.start:self.stop].values
        exog = None if self.exog is None else self.ex_data[self.start:self.stop].values
        armax = ARMA(data,
                     exog=exog,
                     order=(self.p,self.q))
        self.__armax_result = armax.fit(method='mle',
                                        trend='c' if self.has_const else 'nc',
                                        #solver='cg',
                                        #transparams=False,
                                        disp=0)
        #print(self.__armax_result.params['x1'])
        self.fit_params = self.__armax_result.params
        self.const = self.fit_params[0] if self.has_const else 0
        self.exB = self.fit_params[1 if self.has_const else 0:-self.p-self.q]
        self.arP = self.fit_params[-self.p-self.q:-self.q] if self.q > 0 else self.fit_params[-self.p:]
        self.maQ = self.fit_params[-self.q:] if self.q > 0 else []
        #self.arP = self.fit_params[-1]
        #self.exog_params = self.fit_params[1 if self.has_const else 0:-1]
        #print(f'const:{self.const}\nar1:{self.load_param}\nex1:{self.exog_params}')
    
    def predict_one_step_ahead(self,fc_end):
        data = self.load_data[self.stop-timedelta(hours=self.p-1):fc_end].values[:,0]
        exog = self.ex_data[self.stop-timedelta(hours=self.p-1):fc_end].values
        if self.q > 0:
            resid = np.zeros(len(data)-self.p+self.q)
            resid[:self.q] = self.__armax_result.resid[-self.q:]
        pred = np.empty(len(data)-self.p)
        def m_t(index):
            return self.const if self.exog is None else self.const + np.dot(self.exB,exog[index].transpose())
        
        for t in range(len(data)-self.p):
            ar_term = m_t(t+self.p) + np.dot(self.arP[::-1],data[t:t+self.p]-m_t(slice(t,t+self.p)))
            ma_term = 0 if self.q == 0 else np.dot(self.maQ[::-1],resid[t:t+self.q])
            y = ar_term + ma_term
            if self.q > 0:
                resid[t+self.q] = data[t+self.p] - y
            pred[t] = y
        
        fc = TSForecast(1,pred,
                        self.load_data[self.stop+timedelta(hours=1):fc_end].values[:,0],
                        self.start,self.stop,self.stop+timedelta(hours=1),fc_end) 
        self.forecasts.append(fc)
        return fc
    
    def summary(self):
        """Prints summary
        Returns
        -------
        None
        """
        log.info(f'\n{self.__armax_result.summary()}')


#start = datetime(2015,1,8)
#stop = datetime(2017,12,31)
#fc_end = datetime(2018,12,31)
#exog = ['load_lag','t2m_mean','weekend','t2m_top10']
##fc_end = stop + timedelta(days=14)

#armax = ARMAX_forecast(start,stop,2,1,const=True,exog=exog)
#armax.train()
#armax.summary()

#armax.predict_one_step_ahead(fc_end)
#log.info(armax.forecasts[0])
#armax.forecasts[0].__plot__()

#plt.legend()
#plt.show()


class LR_forecast:
    """TODO
    
    """
    def __init__(self,start,stop,frequency=1):
        """TODO
        
        """
        self.start=start
        self.stop=stop
        self.freq=frequency
        self.__lr_result=None
    
    def train(self):
        """TODO
        
        """
        load_reader = LoadReader()
        data = load_reader.vals4slice(de_load,
                                      self.start,
                                      self.stop,
                                      step=self.freq).values
        date_range = pd.date_range(self.start,
                                   self.stop,
                                   freq=f'{self.freq}H').to_pydatetime()
        model = OLS(date_range,data)
        self.__lr_result = model.fit()
    
    def predict(self,hours_ahead):
        """TODO
        
        """
        p = self.__lr_result.predict()
        log.info(len(p))
        log.info(p)


#arma = ARMAResults.load('/home/marcel/Dropbox/data/ARMA_c_p2q2.pkl')
#print(arma.aic)
#print(dir(arma))
#for x in range(10):
    #print(arma.forecast())


#aics = {}
#bics = {}
#hqics = {}
#ar_params = range(1,6)
#ma_params = range(1,6)
#for ar_param in ar_params:
    #for ma_param in ma_params:
        #print(f'training model for p={ar_param} and q={ma_param}')
        #try:
            #model = ARMA(data,order=(ar_param,ma_param),dates=date_range)
            
            #arma_result = model.fit(method='css-mle',
                                    #trend='c',
                                    #transparams=True,
                                    #disp=-1)
            
            #arma_result.save(f"{data_path}ARMA_c_p{ar_param}q{ma_param}.pkl")
            #aics[(ar_param,ma_param)] = arma_result.aic
            #bics[(ar_param,ma_param)] = arma_result.bic
            #hqics[(ar_param,ma_param)] = arma_result.hqic
            #print('done')
        #except Exception as e:
            #print(f'exception occured:\n{e}')

#from operator import add,itemgetter
#sums = []
#for (aic,bic,hqic) in zip(aics.items(),bics.items(),hqics.items()):
    #sums.append((aic[0],aic[1]+bic[1]+hqic[1]))

#print(f'hqics:\n{sorted(hqics.items(),key=itemgetter(1))}')
#print(f'sums:\n{sorted(sums,key=itemgetter(1))}')



#lreg = LR_forecast(tstart, tstop)
#lreg.train()
#lreg.predict(24)

#ldata = LoadReader().vals4slice(de_load,tstart,tstop,1)
#pr = LinearRegression().fit()

#start = datetime.now()
#arima = ARIMA_forecast(datetime(2016,1,1),datetime(2018,5,1),data_column=de_load, p=4, d=0, q=2)
#arima.train()
#arima.aic()
#fname = arima.save()

#print(f'training took {(datetime.now()-start).seconds}s')

##arima.plot_predict(336)

#from matplotlib import pyplot as plt
#start = datetime(2015,1,1)
#stop = datetime(2017,12,31)
#fc_range = 24
#stop_delta = stop + timedelta(hours=fc_range)

#date_range = pd.date_range(start,stop_delta,freq='2H')
#load_reader = LoadReader()
#data = load_reader.vals4slice(de_load,start,stop_delta,step=2)

##arima = ARIMA_forecast()
##arima.load("/home/marcel/Dropbox/data/ARIMA_p4d1q2.pkl")
#forecast_range = pd.date_range(stop,stop_delta,freq='2H')
#pred = arima.predict(fc_range)

#plt.plot(date_range[-fc_range*4:],data[-fc_range*4:],color='b')
#plt.plot(forecast_range,pred,color='r')

#plt.show()

#print(arima.arima_result.aic)

#from statsmodels.graphics.tsaplots import plot_acf
#from statsmodels.tsa.stattools import acf
#from matplotlib import pyplot as plt
#import numpy as np

#start = datetime(2015,1,1)
#stop = datetime(2017,12,31)
#rd = LoadReader()
#data = rd.vals4step(de_load,step=2).interpolate_na(dim='utc_timestamp',method='linear').values
##data = rd.vals4slice(de_load,start,stop,step=2).values

##autocorr = np.correlate(data,data,mode='full')
##plt.plot(autocorr)
#plot_acf(data,fft=True,use_vlines=True,lags=84)

##autocorr = acf(data,missing='drop')
##plt.plot(autocorr)

#plt.show()


    