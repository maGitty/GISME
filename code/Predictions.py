#!/usr/bin/python3
from LoadReader import LoadReader
from WeatherReader import WeatherReader
from glob_vars import de_load,hertz_load,amprion_load,tennet_load,transnet_load,data_path,log

from datetime import datetime,timedelta
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARMA,ARMAResults
from statsmodels.tsa.tsatools import add_trend
from statsmodels.tools.eval_measures import aicc
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle
import sys
import functools
import operator
import os

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class TSForecast:
    """Used to store time series forecast"""
    def __init__(self,hours_ahead,forecast,actual,train_start,train_stop,fc_start,fc_stop):
        """Initializes instance
        Parameters
        ----------
        hours_ahead : int
                      specifies how many hours ahead forecast is done
        forecast    : numpy.ndarray
                      contains forecast data
        actual      : numpy.ndarray
                      contains actual data
        tstart      : datetime.datetime
                      forecast start time
        tstop       : datetime.datetime
                      forecast stop time
        
        Returns
        -------
        None
        """
        assert len(forecast) == len(actual), f'lengths do not equal: {len(forecast)},{len(actual)}'
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
            f"\nmape : {np.round(self.mape(),2)} , rmse : {np.round(self.rmse(),2)}"
        return sum_str
    
    def mape(self):
        """Calculates Mean Average Percentage Error of this forecast
        Returns
        -------
        Mean Average Percentage Error as float
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
            f"to {self.fc_stop} trained from {self.train_start} to {self.train_stop}:"\
            f"\nmape : {np.round(self.mape(),2)} , rmse : {np.round(self.rmse(),2)}"
        
        return sum_str


class ARMA_forecast:
    """ARMA forecast"""
    def __init__(self,start,stop,p=None,q=None):
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
        
        Returns
        -------
        None
        """
        self.load_reader = LoadReader()
        self.p = p
        self.q = q
        self.start = start
        self.stop = stop
        self.__arma_result = None
        self.forecasts = []

    @staticmethod
    def load(fname):
        """Loads ARMA_forecast instance from file name
        Parameters
        ----------
        fname : string
                path to pickled file containing instance
        
        Returns
        -------
        ARMA_forecast instance if exists, raising Exception otherwise
        """
        try:
            return pickle.load(open(fname,'rb'))
        except Exception as e:
            log.exception(f"an error occured:\n{e}")
            raise OSError(f'file not found: {fname}')

    def save(self):
        """Stores instance to pickle
        Returns
        -------
        file name of stored instance as string
        """
        fname = os.path.join(data_path,f"ARMA_p{self.p}q{self.q}.pkl")
        pickle.dump(self,open(fname,'wb'))
        return fname
    
    def train(self):
        """Fits ARMA_forecast instance with specified endogenous data
           Needs to be called before forecasting
        Returns
        -------
        None
        """
        # time steps of values in hours
        data = self.load_reader.vals4slice(de_load,
                                           self.start,
                                           self.stop,
                                           step=1).ffill(dim='utc_timestamp').values
        
        date_range = pd.date_range(self.start,
                                   self.stop,
                                   freq="1H")
        model = ARMA(data,
                      order=(self.p,self.q),
                      dates=date_range)
        self.__arma_result = model.fit(method='css-mle',
                                      trend='c',
                                      transparams=False,
                                      full_output=-1)
        self.fit_params = self.__arma_result.params
    
    def predict_next(self,hours_ahead=24):
        """Predicts for given amount of hours from current trained point
        Parameters
        ----------
        hours_ahead : int
                      number of hours to forecast
        
        Returns
        -------
        predicted values as numpy.ndarray
        """
        predictions = self.__arma_result.predict(self.stop,
                                                self.stop+timedelta(hours=hours_ahead),
                                                dynamic=False)
        return predictions
    
    def predict_range(self,stop_time,hours_ahead):
        """Predicts from instance tstop to given stop_time for every hour with given hours_ahead
        Parameters
        ----------
        stop_time   : datetime.datetime
                      time of last forecast
        hours_ahead : list of int
                      time steps to forecast
        
        Returns
        -------
        list of forecasts for each number in hours_ahead respectively as list
        """
        assert self.__arma_result is not None, "did not train arma yet"
        max_hours = max(hours_ahead)
        delta1h = timedelta(hours=1)
        forecast = [[] for hour in range(len(hours_ahead))]
        
        fc = self.__arma_result.forecast(steps=max_hours)[0]
        
        for i,hours in enumerate(hours_ahead):
            forecast[i].append(fc[hours-1])
        
        stop_counter = self.stop
        while stop_counter < stop_time:
            stop_counter+=delta1h
            data = self.load_reader.vals4slice(de_load,
                                          self.start,
                                          stop_counter,
                                          step=1).ffill(dim='utc_timestamp').values
            date_range = pd.date_range(self.start,stop_counter,freq="1H")
            arma = ARMA(data,order=(self.p,self.q),dates=date_range)
            arma.method = 'css-mle'
            self.__arma_result.initialize(arma, self.fit_params)
            fc = self.__arma_result.forecast(steps=max_hours)[0]
    
            for i,hours in enumerate(hours_ahead):
                forecast[i].append(fc[hours-1])
        
        actual_data = self.load_reader.vals4slice(de_load,self.stop,stop_counter,step=1).ffill(dim='utc_timestamp').values
        for i,fc in enumerate(forecast):
            self.forecasts.append(TSForecast(hours_ahead[i],fc,actual_data,self.start,self.stop,self.stop,stop_counter))
        
        return forecast
    
    def summary(self):
        """Prints summary
        Returns
        -------
        None
        """
        log.info(f'\n{self.__arma_result.summary()}')


class ARMAX_forecast:
    """ARMAX forecast"""
    def __init__(self,start,stop,p=None,q=None,exog=None):
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
                used exogenous variables
        
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
        self.method = 'css-mle'
    
    def __load_exog(self,tstart=None,tstop=None,range_start=None):
        """loads exogenous variables as numpy.ndarray based on self.exog
           possible choices:
             - 'dayofweek'    : number of week day
             - 'data_counter' : counting data points beginning from 0
             - 't2m_max'      : maximum of 2 metre temperature for each step
             - 't2m_mean'     : mean of 2 metre temperature for each step
             - 't2m_all'      : every grid point of 2 metre temperature as single exogenous variable
             - 'u10_all'      : every grid point of 10 metre U wind component as single exogenous variable
             - 'v10_all'      : every grid point of 10 metre V wind component as single exogenous variable
             - 'lai_hv_all'   : every grid point of high vegetation leaf area index as single exogenous variable
             - 'lai_lv_all'   : every grid point of low vegetation leaf area index as single exogenous variable
             - 'lcc_all'      : every grid point of low cloud cover as single exogenous variable
             - 'stl1_all'     : every grid point of soil temperature level 1 as single exogenous variable
             - 'slhf_all'     : every grid point of surface latent heat flux as single exogenous variable
             - 'str_all'      : every grid point of surface net thermal radiation as single exogenous variable
             - 'sshf_all'     : every grid point of surface sensible heat flux as single exogenous variable
             - 'tcc_all'      : every grid point of total cloud cover as single exogenous variable
             - 'tcrw_all'     : every grid point of total column rain water as single exogenous variable
             - 'fdir_all'     : every grid point of total sky direct solar radiation at surface as single exogenous variable
        
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
            return None
        exog = []
        date_range = pd.date_range(tstart,tstop,freq="1H")
        dow = date_range.to_series().dt.dayofweek.values
        if 'dayofweek' in self.exog:
            # extend exog by 7 dummy arrays, each for one weekday
            cat = np.zeros((7,dow.size))
            for i in range(7):
                cat[i,dow==i] = 1
            exog.extend(cat)
        if 'weekend' in self.exog:
            wend = np.zeros((dow.size))
            wend[dow==5] = 1
            wend[dow==6] = 1
            exog.append(wend)
        if 'data_counter' in self.exog:
            if range_start is None:
                exog.append(np.array(range(date_range.size)))
            else:
                exog.append(np.array(range(range_start,range_start+date_range.size)))
        if 'hour' in self.exog:
            exog.append(date_range.to_series().dt.hour.values)
        if 't2m_max' in self.exog:
            exog.append(self.weather_reader.max_slice('t2m',tstart,tstop))
        if 't2m_mean' in self.exog:
            exog.append(self.weather_reader.mean_slice('t2m',tstart,tstop))
        if 't2m_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('t2m',tstart,tstop))
        if 'u10_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('u10',tstart,tstop))
        if 'v10_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('v10',tstart,tstop))
        if 'lai_hv_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('lai_hv',tstart,tstop))
        if 'lai_lv_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('lai_lv',tstart,tstop))
        if 'lcc_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('lcc',tstart,tstop))
        if 'stl1_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('stl1',tstart,tstop))
        if 'slhf_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('slhf',tstart,tstop))
        if 'str_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('str',tstart,tstop))
        if 'sshf_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('sshf',tstart,tstop))
        if 'tcc_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('tcc',tstart,tstop))
        if 'tcrw_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('tcrw',tstart,tstop))
        if 'fdir_all' in self.exog:
            exog.extend(self.weather_reader.flattened_slice('fdir',tstart,tstop))
        
        return np.column_stack(exog)
    
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
        try:
            return pickle.load(open(fname,'rb'))
        except Exception as e:
            log.exception(f"An error occured:\n{e}")
            raise OSError(f'file not found: {fname}')
    
    def save(self):
        """Stores instance to pickle
        Returns
        -------
        file name of stored instance as string
        """
        fname = os.path.join(data_path,f"ARMAX_p{self.p}q{self.q}.pkl")
        pickle.dump(self,open(fname,'wb'))
        return fname
    
    def train(self):
        """Fits ARMAX_forecast instance with specified endogenous and exogenous data
           Needs to be called before forecasting
        Returnsobject
        -------
        None
        """
        data = self.load_reader.vals4slice(de_load,
                                           self.start,
                                           self.stop,
                                           step=1).values
        model_exog = self.__load_exog(self.start,self.stop)
        arma = ARMA(data,exog=model_exog,
                     order=(self.p,self.q))
        self.__armax_result = arma.fit(method=self.method,
                                        trend='nc',
                                        transparams=False,
                                        full_output=-1)
        self.fit_params = self.__armax_result.params
    
    def predict_next(self,hours_ahead=24):
        """Predicts for given amount of hours from current trained point
        Parameters
        ----------
        hours_ahead : int
                      number of hours to forecast
        
        Returns
        -------
        predicted values as numpy.ndarray
        """
        predictions = self.__armax_result.forecast(steps=hours_ahead,
                                                   exog=pd.date_range(self.stop+timedelta(hours=1),
                                                                      self.stop+timedelta(hours=hours_ahead),
                                                                      freq='1H').to_series().dt.dayofweek)[0]
        return predictions
    
    def predict_range(self,stop_time,hours_ahead):
        """Predicts from instance tstop to given stop_time for every hour with given hours_ahead
        Parameters
        ----------
        stop_time   : datetime.datetime
                      time of last forecast
        hours_ahead : list of int
                      time steps to forecast
        
        Returns
        -------
        list of forecasts for each number in hours_ahead respectively as list
        """
        assert self.__armax_result is not None, "did not train arma yet"
        max_hours = max(hours_ahead)
        data = self.load_reader.vals4slice(de_load,self.start,self.stop,step=1)
        delta1h = timedelta(hours=1)
        out_of_sample_exog = self.__load_exog(self.stop+delta1h,
                                              self.stop+timedelta(hours=max_hours),len(data))
        fc = self.__armax_result.forecast(steps=max_hours,exog=out_of_sample_exog)[0]
        
        forecast = [[] for hour in range(len(hours_ahead))]
        for i,hours in enumerate(hours_ahead):
            forecast[i].append(fc[hours-1])
        
        stop_counter = self.stop
        while stop_counter < stop_time:
            stop_counter+=delta1h
            data = self.load_reader.vals4slice(de_load,self.start,stop_counter,
                                              step=1).ffill(dim='utc_timestamp').values
            model_exog = self.__load_exog(self.start,stop_counter)
            arma = ARMA(data,exog=model_exog,
                        order=(self.p,self.q))
            arma.method = self.method
            #if self.exog is not None: # needed if constant is included -> did result in higher error
                #arma.exog = add_trend(model_exog, trend='c', prepend=True, has_constant='raise')
            self.__armax_result.initialize(arma,self.fit_params)
            out_of_sample_exog = self.__load_exog(stop_counter+delta1h,
                                                  stop_counter+timedelta(hours=max_hours),
                                                  len(data))
            fc = self.__armax_result.forecast(steps=max_hours,exog=out_of_sample_exog)[0]
    
            for i,hours in enumerate(hours_ahead):
                forecast[i].append(fc[hours-1])
        
        actual_data = self.load_reader.vals4slice(de_load,self.stop,stop_counter,step=1).ffill(dim='utc_timestamp').values
        for i,fc in enumerate(forecast):
            self.forecasts.append(TSForecast(hours_ahead[i],fc,actual_data,self.start,self.stop,self.stop,stop_counter))
        
        return forecast
    
    def summary(self):
        """Prints summary
        Returns
        -------
        None
        """
        log.info(f'\n{self.__armax_result.summary()}')


#def auto_arma(tstart,tstop,ar_stop,ma_stop):
    #load_reader = LoadReader()
    #data = load_reader.vals4slice(de_load,tstart,tstop,step=1).ffill(dim='utc_timestamp').values
    #date_range = pd.date_range(tstart,tstop,freq="1H")
    
    #if not os.path.exists(os.path.join(data_path,f'tstart{tstart.year}_tstop{tstop.year}')):
        #os.mkdir(os.path.join(data_path,f'tstart{tstart.year}_tstop{tstop.year}'))
    
    #for q in range(ma_stop):
        #for p in range(ar_stop):
            #if os.path.exists(os.path.join(data_path,f'tstart{tstart.year}_tstop{tstop.year}',f'ARMA_p{p}q{q}.pkl')):
                #break
            #start = datetime.now()
            #arma = ARMA(data,order=(p,q),dates=date_range)
            #try:
                #arma_result = arma.fit(trend='c',method='css-mle',transparams=True,full_output=-1)
                #arma_result.save(os.path.join(data_path,f'tstart{tstart.year}_tstop{tstop.year}',f'ARMA_p{p}q{q}.pkl'))
                #print(f'trained and saved arma with p={p} and q={q}, took {(datetime.now()-start).seconds}s')
            #except Exception as e:
                #print(f'could not train ARMA({p},{q}):\n{e}')

def print_armas(tstart,tstop,ar_stop,ma_stop):
    pqs = functools.reduce(operator.add,[[(x+1,x),(x,x+1),(x+1,x+1)] for x in range(8)],[])
    
    for q in range(ma_stop):
        for p in range(ar_stop):
            try:
                arma_result = ARMAResults.load(os.path.join(data_path,f'tstart{tstart.year}_tstop{tstop.year}',f'ARMA_p{p}q{q}.pkl'))
                log.info(f'({p}|{q}) -\taic:{np.round(arma_result.aic,1)}\tbic:{np.round(arma_result.bic,1)}\thqic:{np.round(arma_result.hqic,1)}\tresid:{np.array(arma_result.resid).mean().round(3)}\tloglike:{np.round(arma_result.llf,1)}')
            except:
                pass

#tstart = datetime(2017,1,1,0)
#tstop = datetime(2018,1,1,0)
##print_armas(tstart, tstop,6,5)
#armax = ARMAX_forecast(tstart,tstop,1,0,exog=['dayofweek'])
#armax.train()
#armax.summary()
#armax.predict_range(tstop+timedelta(weeks=1),[1,6,24])
#for fc in armax.forecasts:
    #plt.plot(fc.forecast,label=f'{fc.hours_ahead}H')
    #print(fc.mape(), fc.rmse())
#plt.plot(armax.forecasts[0].actual,label='actual')
#plt.legend()
#plt.show()

#arma = ARMA_forecast(tstart,tstop,1,1)
#arma.train()
#fname = arma.save()
#arma = ARMA_forecast.load(fname)
#arma.predict_range(tstop+timedelta(weeks=1), [1,6,24])
#for fc in arma.forecasts:
    #plt.plot(fc.forecast,label=f'{fc.hours_ahead}H')
    #print(fc.mape(), fc.rmse())
#plt.plot(arma.forecasts[0].actual,label='actual')
#plt.legend()
#plt.show()
#ar = ARMAResults.load(os.path.join(data_path,f'tstart2017_tstop2018',f'ARMA_p4q3.pkl'))
#plt.plot(ar.resid)
#plt.show()
#load_reader = LoadReader()
#data = load_reader.vals4slice(de_load,tstart,tstop,step=1).ffill(dim='utc_timestamp').values
#date_range = pd.date_range(tstart,tstop,freq="1H")

#pq = (1,1)
#print('training arma')
#arma = ARMA(data,order=pq,dates=date_range)
#arma_result = arma.fit(trend='c',
                       #method="css-mle",
                       #transparams=True,
                       #full_output=-1)
#arma_result.save(os.path.join(data_path,'arma11.pkl'))
#arma_result = ARMAResults.load(os.path.join(data_path,'arma11.pkl'))
#print(arma_result.summary())
#pickle.dump(arma_result.summary(),open(os.path.join(data_path,'arma11.pkl'),'wb'))
#summ = pickle.load(open(os.path.join(data_path,'arma11.pkl'),'rb'))
#print(summ)
#fit_params = arma_result.params
#fc = arma_result.forecast(steps=24)[0]
#forecast1h = [fc[0]]
#forecast6h = [fc[5]]
#forecast24h = [fc[23]]

#stop_time = tstop
#end_time = datetime(2018,1,7,23)
#delta1h = timedelta(hours=1)
#while stop_time < end_time:
    #stop_time = stop_time+delta1h
    #if stop_time.day==1 and stop_time.hour==0:
        #print(f'forecasting for {stop_time}',end='\r',flush=True)
    #data = load_reader.vals4slice(de_load,tstart,stop_time,step=1).ffill(dim='utc_timestamp').values
    #date_range = pd.date_range(tstart,stop_time,freq="1H")
    #arma = ARMA(data,order=pq,dates=date_range)
    #arma.method = 'css-mle'
    #arma_result.initialize(arma, fit_params)
    #fc = arma_result.forecast(steps=24)[0]
    #forecast1h.append(fc[0])
    #forecast6h.append(fc[5])
    #forecast24h.append(fc[23])

    
#tstart = datetime(2015,1,1,0)
#tstop = datetime(2017,12,31,23)
#end_time = datetime(2018,1,7,23)

#from matplotlib import pyplot as plt

#fc_range = pd.date_range(tstop,end_time,freq='1H')
#data_range = pd.date_range(tstop,end_time,freq='1H')
#data = load_reader.vals4slice(de_load,tstop,end_time,step=1).ffill(dim='utc_timestamp').values
#mape1h = np.mean(np.abs((data - forecast1h) / data)) * 100
#mape6h = np.mean(np.abs((data - forecast6h) / data)) * 100
#mape24h = np.mean(np.abs((data - forecast24h) / data)) * 100
#print(f'MAPE for 1H forecast: {mape1h}')
#print(f'MAPE for 6H forecast: {mape6h}')
#print(f'MAPE for 24H forecast: {mape24h}')

#plt.plot(fc_range, forecast1h, label='1H forecast')
#plt.plot(fc_range, forecast6h, label='6H forecast')
#plt.plot(fc_range, forecast24h, label='24H forecast')
#plt.plot(data_range,data, label='actual value')

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


    