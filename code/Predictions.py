#!/usr/bin/python3
from LoadReader import LoadReader
from glob_vars import de_load,hertz_load,amprion_load,tennet_load,transnet_load,data_path

from datetime import datetime,timedelta
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA,ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX,SARIMAXResults
from statsmodels.tools.eval_measures import aicc
import pandas as pd

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class ARIMA_forecast:
    """TODO
    
    """
    def __init__(self,data_column=None,p=None,d=None,q=None):
        """TODO
        
        """
        self.p = p
        self.d = d
        self.q = q
        self.data_column = data_column
        self.start = datetime(2015,1,1)
        self.stop = datetime(2017,12,31)        
        self.arima_result = None

    def load(self,fname):
        """TODO
        
        """
        try:
            self.arima_result = ARIMAResults.load(fname)
        except:
            print("could not load arima")

    def save(self):
        """TODO
        
        """
        assert self.arima_result is not None, "did not train arima yet"
        
        fname = data_path+f"ARIMA_p{self.p}d{self.d}q{self.q}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        self.arima_result.save(fname)
        return fname
    
    def train(self):
        """TODO
        
        """
        load_reader = LoadReader()
        # time steps of values
        frequency = 2
        data = load_reader.vals4slice(self.data_column,
                                      self.start,
                                      self.stop,
                                      step=frequency).values
        
        date_range = pd.date_range(self.start,
                                   self.stop,
                                   freq=f"{frequency}H")
        model = ARIMA(data,
                      order=(self.p,self.d,self.q),
                      dates=date_range)
        self.arima_result = model.fit(transparams=True,
                                      #maxiter=100,
                                      disp=5,
                                      method='mle')
    
    def predict(self,hours_ahead=24):
        """TODO
        
        """
        predictions = self.arima_result.predict(self.stop,
                                                self.stop+timedelta(hours=hours_ahead),
                                                dynamic=True)
        return predictions
    
    def plot_predict(self,hours_ahead):
        """TODO
        
        """
        self.arima_result.plot_predict(self.stop,
                                       self.stop+timedelta(hours=hours_ahead),
                                       dynamic=True,
                                       plot_insample=False)
        plt.show()
    
    def wald_test_terms(self):
        """TODO
        
        """
        print(self.arima_result.aic())


class SARIMA_forecast:
    """TODO
    
    """
    def __init__(self,data_column=None,p=None,d=None,q=None,P=None,D=None,Q=None,s=84):
        """TODO
        
        """
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.data_column = data_column
        self.start = datetime(2015,1,1)
        self.stop = datetime(2017,12,31)        
        self.sarimax_result = None
    
    def load(self,fname):
        """TODO
        
        """
        try:
            self.sarimax_result = SARIMAXResults.load(fname)
        except:
            print("could not load arima")

    def save(self):
        """TODO
        
        """
        assert self.sarimax_result is not None, "did not train arima yet"
        
        fname = data_path+f"SARIMA_p{self.p}d{self.d}q{self.q}P{self.P}D{self.D}Q{self.Q}s{self.s}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        self.sarimax_result.save(fname)
        return fname
    
    def train(self):
        """TODO
        
        """
        load_reader = LoadReader()
        # time steps of values
        frequency = 2
        data = load_reader.vals4slice(self.data_column,
                                      self.start,
                                      self.stop,
                                      step=frequency).values
        
        date_range = pd.date_range(self.start,
                                   self.stop,
                                   freq=f"{frequency}H")
        model = SARIMAX(data,
                      order=(self.p,self.d,self.q),
                      seasonal_order=(self.P,self.D,self.Q,self.s),
                      dates=date_range)
        self.sarimax_result = model.fit()
        print(self.sarimax_result)


class ARIMAX:
    """TODO
    
    """
    def __init__(self):
        """TODO
        
        """
        pass

sarima = SARIMA_forecast(data_column=de_load,p=2,d=1,q=2,P=1,D=0,Q=1)
sarima.train()

#arima = ARIMA_forecast(data_column=de_load, p=12, d=0, q=2)
#arima.train()
#fname = arima.save()

#arima.plot_predict(336)

#from matplotlib import pyplot as plt
#start = datetime(2015,1,1)
#stop = datetime(2017,12,31)
#fc_range = 336
#stop_delta = stop + timedelta(hours=fc_range)

#date_range = pd.date_range(start,stop_delta,freq='2H')
#load_reader = LoadReader()
#data = load_reader.vals4slice(de_load,start,stop_delta,step=2)

#arima = ARIMA_forecast()
#arima.load("/home/marcel/Dropbox/data/ARIMA_p12d0q2_20190722_1547.pkl")
#forecast_range = pd.date_range(stop,stop_delta,freq='2H')
#pred = arima.predict(fc_range)

#plt.plot(date_range[-fc_range:],data[-fc_range:],color='b')
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


    