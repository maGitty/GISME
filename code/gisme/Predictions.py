#!/usr/bin/python3

"""
In this module, the different methods for forecasting are implemented
"""

from gisme import (de_load, data_path, log, data_max_date,numpy_funcs)
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
    """Used to store time series forecast
    
    Attributes
    ----------
    date_range  : DatetimeIndex
                  the date range of the forecast from fc_start to fc_stop
    hours_ahead : integer
                  specifies the number of hours the forecast predicts ahead (currently only 1h possible)
    forecast    : numpy.ndarray
                  the array of forecasted values
    actual      : numpy.ndarray
                  the array of actual values
    train_start : datetime.datetime
                  start time of training
    train_stop  : datetime.datetime
                  end time of training
    fc_start    : datetime.datetime
                  start time of forecast
    fc_stop     : datetime.datetime
                  end time of forecast
    """
    def __init__(self, hours_ahead, forecast, actual, train_start,
                 train_stop, fc_start, fc_stop):
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
            f"\nrmse : {np.round(self.rmse(), 3):.3f} , mae : {np.round(self.mae(), 3):.3f} ,"\
            f" mpe : {np.round(self.mpe(), 3):.3f} , mape : {np.round(self.mape(), 3):.3f}"
        return sum_str
    
    def __teXstr__(self):
        """Returns the error measures of this forecast
        
        Returns
        -------
        The error measures as string in tex format to insert into a table only missing the type of the model
        """
        return f"{np.round(self.rmse(), 3):.3f} & {np.round(self.mae(), 3):.3f} & "\
               f"{np.round(self.mpe(), 3):.3f} & {np.round(self.mape(), 3):.3f}\\\\"
    
    def __plot__(self):
        """Plots the respective forecast
        
        Still needs to call matplotlib.pyplot.show() which is omitted
        here in case another forecast should be plotted in the same figure
           
        Returns
        -------
        None
        """
        date_range = pd.date_range(self.fc_start, self.fc_stop, freq='1H')
        plt.plot(date_range, self.actual, label='actual')
        plt.plot(date_range, self.forecast, label=f'{self.hours_ahead}H forecast')
        plt.legend()
    
    def sel(self, start, stop):
        """Return a pandas.DataFrame containing the actual and forecast data from start to stop
        
        Parameters
        ----------
        start : datetime.datetime
                start time
        stop  : datetime.datetime
                stop time
        
        Returns
        -------
        pandas.DataFrame containing actual and forecast values
        """
        df = pd.DataFrame(data={'actual': self.actual, 'forecast': self.forecast},
                          index=self.date_range)
        return df[start:stop]
    
    def difference(self, other_fc):
        """Return difference between values of this forecast and other_fc
        
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
            f"\nrmse : {np.round(self.rmse(), 3):.3f} , mae : {np.round(self.mae(), 3):.3f} ,"\
            f" mpe : {np.round(self.mpe(), 3):.3f} , mape : {np.round(self.mape(), 3):.3f}"
        
        return sum_str


class ARMAXForecast:
    """This class provides training and forecasting methods for ARMAX models
    
    Attributes
    ----------
    load_reader      : LoadReader
                       used to load the load data
    weather_reader   : WeatherReader
                       used to load the weather data
    p                : integer
                       the number of AR terms in the ARMA
    q                : integer
                       the number of MA terms in the ARMA
    start            : datetime.datetime
                       the start time of the training
    stop             : datetime.datetime
                       the end time of the training
    forecasts        : list of TSForecast
                       the forecasts performed with this ARMAXForecast instance
    exog             : list of strings
                       names of exogenous variables used as exogenous inputs
    has_const        : boolean
                       whether to use a constant or not
    load_data        : numpy.array
                       the array of load data used for training and forecasting
    ex_data          : numpy.ndarray
                       the N-dimensional array used as exogenous inputs for training and forecasting
    const            : float
                       the models constant, if no constant is included, this is just 0
    exB              : numpy.array
                       the coefficients for the exogenous inputs
    arP              : numpy.array
                       the coefficients for the autoregressive terms
    maQ              : numpy.array
                       the coefficients for the moving average terms
    __armax_result__ : statsmodels.tsa.arima_model.ARMAResults
                       the return value of the ARMA.fit method that contains the training results
    """
    def __init__(self, start, stop, p=None, q=None, exog=None, const=True):
        """Initializes instance
        
        possible choices for exogenous variables in @exog:
          * 'load_lag'     : load data shifted by one week (with this parameter only data from 2015/01/08 can be used)
          * 'dayofweek'    : week day (one dummy per day)
          * 'weekend'      : dummy for whether it is weekend or not (1 or 0)
          * 'data_counter' : counting data points beginning from 0
          * combinations of weather variables and different types, joined with underscore (eg 't2m_mean', 'tcc_all'): 
            weather variables:
            ['u10', 'v10', 't2m', 'lai_hv', 'lai_lv', 'lcc',
            'stl1', 'slhf', 'str', 'sshf', 'tcc', 'tcrw', 'fdir']
            types:
            ['min', 'max', 'mean', 'top10', 'all']
            about types:
              * 'min' reduces the data over longitude and latitude using the min value for each time step,
              * 'max' reduces the data over longitude and latitude using the max value for each time step,
              * 'mean' reduces the data over longitude and latitude using the average for each time step,
              * 'top10' takes the grid points of the 10 regions with the highest population
              * 'all' just adds each single grid point as an exogenous parameter
        
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
        self.__armax_result__ = None
        self.forecasts = []
        self.exog = None if exog is [] else exog
        self.has_const = const
        self._load_data()
    
    @staticmethod
    def load(fname):
        """Loads ARMAXForecast instance from file name
        
        Parameters
        ----------
        fname : string
                path to pickled file containing instance
        
        Returns
        -------
        ARMAXForecast instance if exists, raising Exception otherwise
        """
        return pickle.load(open(fname, 'rb'))
    
    def save(self):
        """Stores instance to pickle
        
        Returns
        -------
        file name of stored instance as string
        """
        dir_pth = os.path.join(data_path, 'ARMAX')
        if not os.path.exists(dir_pth):
            os.mkdir(dir_pth)
        fname = os.path.join(dir_pth,
                             f'start{self.start.strftime("%Y%m%d%H")}_stop'\
                             f'{self.stop.strftime("%Y%m%d%H")}_p{self.p}q{self.q}'\
                             f'{"" if self.exog is None else "_"+"_".join(self.exog)}.pkl')
        log.info(f'saving armax as {fname}')
        pickle.dump(self, open(fname,'wb'))
        return fname
    
    def _get_exog(self,var_name):
        """Private method that adds single exogenous variables to the exogenous data
        
        Parameters
        ----------
        var_name : string
                   name of the variable to add to exogenous data
        
        Returns
        -------
        None
        """
        if var_name is 'load_lag':
            delta1week = timedelta(weeks=1)
            self.ex_data['load_shift'] = self.load_reader.vals4slice(de_load, self.start-delta1week,
                                                                     data_max_date-delta1week, step=1)
        elif var_name is 'dayofweek':
            dow = self.ex_data.index.to_series().dt.dayofweek
            self.ex_data['Monday'] = (dow == 0).astype(int)
            self.ex_data['Tuesday'] = (dow == 1).astype(int)
            self.ex_data['Wednesday'] = (dow == 2).astype(int)
            self.ex_data['Thursday'] = (dow == 3).astype(int)
            self.ex_data['Friday'] = (dow == 4).astype(int)
            self.ex_data['Saturday'] = (dow == 5).astype(int)
            self.ex_data['Sunday'] = (dow == 6).astype(int) 
        elif var_name is 'weekend':
            dow = self.ex_data.index.to_series().dt.dayofweek            
            self.ex_data['weekend'] = dow.isin([5, 6]).astype(int)
        elif var_name is 'data_counter':
            self.ex_data['data_counter'] = range(len(self.load_data))
        else:
            var = var_name.split('_')
            var_type = var[-1]
            var = '_'.join(var[:-1]) if len(var) > 2 else var[0]
            if var in self.weather_reader.get_vars():
                if var_type == 'top10':
                    data = self.weather_reader.demography_top_n_regions4timeslice(var, self.start, data_max_date, 10, 2018)
                    for i, top_i in enumerate(data):
                        self.ex_data[f'{var}{i}gridpoint'] = top_i
                elif var_type == 'all':
                    data = self.weather_reader.stackedvals4timeslice(var, self.start, data_max_date)
                    for i, gridpoint in enumerate(data):
                        self.ex_data[f'{var}{i}'] = gridpoint
                else:
                    numpy_func = numpy_funcs[var_type]
                    self.ex_data[f'{var}_{var_type}'] = self.weather_reader.vals4timeslice_reduced(var, self.start,
                                                                                                   data_max_date, numpy_func)  
            else:
                raise NameError(f'variable name not found: {var_name}')
        #elif var_name.split('_')[0] in self.weather_reader.get_vars():
            #var = var_name.split('_')[0]
            #var_type = var_name.split('_')[1]
            #if var_type == 'top10':
                #data = self.weather_reader.demography_top_n_regions4timeslice(var, self.start, data_max_date, 10, 2018)
                #for i, top_i in enumerate(data):
                    #self.ex_data[f'{var}{i}gridpoint'] = top_i
            #elif var_type == 'all':
                #data = self.weather_reader.stackedvals4timeslice(var, self.start, data_max_date)
                #for i, gridpoint in enumerate(data):
                    #self.ex_data[f'{var}{i}'] = gridpoint
            #else:
                #numpy_func = numpy_funcs[var_type]
                #self.ex_data[f'{var}_{var_type}'] = self.weather_reader.vals4timeslice_reduced(var, self.start,
                                                                                               #data_max_date, numpy_func)                
        #else:
            #raise NameError(f'variable name not found: {var_name}')
    
    def _load_data(self):
        """Private method that loads the load data, but also exogenous variables as pandas.DataFrame based on strings in self.exog
        
        Returns
        -------
        None
        """
        load_data = self.load_reader.vals4slice(de_load, self.start, data_max_date, step=1)
        date_range = pd.date_range(self.start, data_max_date, freq='1H')
        self.load_data = pd.DataFrame(data={'load': load_data}, index=date_range)
        self.ex_data = pd.DataFrame(data=[], index=date_range)
        
        if self.exog is None:
            return
        for var_name in self.exog:
            self._get_exog(var_name)
    
    def train(self):
        """Fits ARMAXForecast instance with specified endogenous and exogenous data
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
        self.__armax_result__ = armax.fit(method='mle',
                                        trend='c' if self.has_const else 'nc',
                                        #solver='cg',
                                        #transparams=False,
                                        disp=0)
        #print(self.__armax_result__.params['x1'])
        fit_params = self.__armax_result__.params
        self.const = fit_params[0] if self.has_const else 0
        self.exB = fit_params[1 if self.has_const else 0:-self.p-self.q]
        self.arP = fit_params[-self.p-self.q:-self.q] if self.q > 0 else fit_params[-self.p:]
        self.maQ = fit_params[-self.q:] if self.q > 0 else []
    
    def predict_one_step_ahead(self,fc_end):
        """Predicts one step ahead from instance stop time to given fc_end for every hour
        
        Parameters 
        ---------- 
        fc_end : datetime.datetime 
                 time of last forecast 
         
        Returns 
        -------
        TSForecast instance containing actual and forecast values
        """
        data = self.load_data[self.stop-timedelta(hours=self.p-1):fc_end].values[:,0]
        exog = self.ex_data[self.stop-timedelta(hours=self.p-1):fc_end].values
        if self.q > 0:
            resid = np.zeros(len(data)-self.p+self.q)
            resid[:self.q] = self.__armax_result__.resid[-self.q:]
        pred = np.empty(len(data)-self.p)
        def m_t(index):
            return self.const if self.exog is None else self.const + np.dot(self.exB, exog[index].transpose())
        
        for t in range(len(data)-self.p):
            ar_term = m_t(t+self.p) + np.dot(self.arP[::-1],data[t:t+self.p]-m_t(slice(t, t+self.p)))
            ma_term = 0 if self.q == 0 else np.dot(self.maQ[::-1], resid[t:t+self.q])
            y = ar_term + ma_term
            if self.q > 0:
                resid[t+self.q] = data[t+self.p] - y
            pred[t] = y
        
        fc = TSForecast(1, pred, self.load_data[self.stop+timedelta(hours=1):fc_end].values[:,0],
                        self.start, self.stop, self.stop+timedelta(hours=1), fc_end) 
        self.forecasts.append(fc)
        return fc
    
    def summary(self):
        """Prints summary
        
        Returns
        -------
        None
        """
        log.info(f'\n{self.__armax_result__.summary()}')

