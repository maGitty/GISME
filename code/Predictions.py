#!/usr/bin/python3
from LoadReader import LoadReader
from glob_vars import de_load,hertz_load,amprion_load,tennet_load,transnet_load,data_path

from datetime import datetime,timedelta
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA,ARIMAResults,ARMA,ARMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX,SARIMAXResults
from statsmodels.tools.eval_measures import aicc
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import sys

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class TSForecast:
    """TODO
    
    """
    def __init__(self,forecast,actual,tstart,tstop):
        """TODO
        
        """
        assert len(forecast) == len(actual)
        self.forecast = forecast
        self.actual = actual
        self.start = tstart
        self.stop = tstop


class ARIMA_forecast:
    """TODO
    
    """
    def __init__(self,start,stop,data_column=None,p=None,d=None,q=None):
        """TODO
        
        """
        self.p = p
        self.d = d
        self.q = q
        self.data_column = data_column
        self.start = start
        self.stop = stop
        self.__arima_result = None

    def load(self,fname):
        """TODO
        
        """
        try:
            self.__arima_result = ARIMAResults.load(fname)
        except:
            print("could not load arima")

    def save(self):
        """TODO
        
        """
        assert self.__arima_result is not None, "did not train arima yet"
        
        fname = data_path+f"ARIMA_p{self.p}d{self.d}q{self.q}.pkl"
        self.__arima_result.save(fname)
        return fname
    
    def train(self):
        """TODO
        
        """
        load_reader = LoadReader()
        # time steps of values in hours
        frequency = 1
        data = load_reader.vals4slice(self.data_column,
                                      self.start,
                                      self.stop,
                                      step=frequency).ffill(dim='utc_timestamp').values
        
        date_range = pd.date_range(self.start,
                                   self.stop,
                                   freq=f"{frequency}H")
        model = ARIMA(data,
                      order=(self.p,self.d,self.q),
                      dates=date_range)
        self.__arima_result = model.fit(disp=5,
                                      method='mle',
                                      trend='nc')
    
    def predict(self,hours_ahead=24):
        """TODO
        
        """
        predictions = self.__arima_result.predict(self.stop,
                                                self.stop+timedelta(hours=hours_ahead),
                                                dynamic=False)
        return predictions
    
    def plot_predict(self,start,hours_ahead):
        """TODO
        
        """
        self.__arima_result.plot_predict(start,
                                       start+timedelta(hours=hours_ahead),
                                       dynamic=True,
                                       plot_insample=False)
        plt.show()
    
    def aic(self):
        """TODO
        
        """
        print(self.__arima_result.summary())


class ARMA_forecast:
    """TODO
    
    """
    def __init__(self,start,stop,p=None,q=None):
        """TODO
        
        """
        self.load_reader = LoadReader()
        self.p = p
        self.q = q
        self.start = start
        self.stop = stop
        self.__arma_result = None

    def load(self,fname):
        """TODO
        
        """
        try:
            self.__arma_result = ARMAResults.load(fname)
        except:
            print("could not load arma")

    def save(self):
        """TODO
        
        """
        assert self.__arma_result is not None, "did not train arma yet"
        
        fname = f"{data_path}ARMA_p{self.p}q{self.q}.pkl"
        self.__arma_result.save(fname)
        return fname
    
    def train(self):
        """TODO
        
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
                                      transparams=True,
                                      full_output=-1)
        self.fit_params = self.__arma_result.params
    
    def predict_next(self,hours_ahead=24):
        """TODO
        
        """
        predictions = self.__arma_result.predict(self.stop,
                                                self.stop+timedelta(hours=hours_ahead),
                                                dynamic=False)
        return predictions
    
    def predict_range(self,stop_time,hours_ahead):
        """TODO
        
        """
        assert self.__arma_result is not None, "did not train arma yet"
        max_hours = max(hours_ahead) if type(hours_ahead) is list else hours_ahead
        delta1h = timedelta(hours=1)
        forecast = [[] for hour in range(len(hours_ahead))]
        
        fc = self.__arma_result.forecast(steps=max_hours)[0]
        
        if type(hours_ahead) is list:
            for i,hours in enumerate(hours_ahead):
                forecast[i].append(fc[hours-1])
        else:
            forecast.append(fc[hours_ahead-1])
        
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
    
            if type(hours_ahead) is list:
                for i,hours in enumerate(hours_ahead):
                    forecast[i].append(fc[hours-1])
            else:
                forecast.append(fc[hours_ahead-1])
        return forecast
    
    def summary(self):
        """TODO
        
        """
        print(self.__arma_result.summary())


#tstart = datetime(2015,1,1,0)
#tstop = datetime(2017,12,31,23)
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
        print(len(p))
        print(p)


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


class ARIMAX:
    """TODO
    
    """
    def __init__(self):
        """TODO
        
        """
        pass


    