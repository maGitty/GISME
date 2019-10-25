#!/usr/bin/python3

"""
After having installed all required packages, using the provided requirements.txt,
this script allows executing the methods implemented in the gisme package.
"""

# module imports
from gisme import (data_path, de_load, demography_file, log)
from gisme.DataPlotter import DataPlotter
from gisme.LoadReader import LoadReader
from gisme.Predictions import ARMAXForecast,TSForecast
from gisme.Utility import Utility
from gisme.WeatherReader import WeatherReader

# external imports
from matplotlib import pyplot as plt
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import os
import sys
import holidays
import itertools
from statsmodels.tsa.arima_model import ARMA

#top10vars = ['t2m_top10','u10_top10','v10_top10','lai_hv_top10',
             #'lai_lv_top10','lcc_top10','stl1_top10','slhf_top10',
             #'str_top10','sshf_top10','tcc_top10','tcrw_top10','fdir_top10']

#start = datetime(2015,1,8)
#stop = datetime(2017,12,31)
#fc_end = datetime(2018,12,31)

#pl = DataPlotter(fmt='pdf', save=True, show=True, isin=True)  # , shape=(2, 2))
#pl.plot_isin_top_n(100,2018)

#t_start = datetime(2015, 1, 8)
#t_stop = datetime(2017, 12, 31)
#end = datetime(2018, 12, 31)

#plot_start = datetime(2018, 1, 1)
#plot_end = datetime(2018, 1, 8)
#pl.plot_armax_forecast(t_start, t_stop, end, 2, 2, exog=['t2m_max'])
#pl.plot_armax_forecast(t_start, t_stop, end, 2, 2, exog=['t2m_mean'])

#for exog in top10vars:
    #print(exog)
    #try:
        #pl.plot_armax_forecast(t_start, t_stop, end, 2, 2, exog=[exog], plot_range=(plot_start, plot_end))
    #except Exception as e:
        #print(str(e))

#ut = Utility()
#print(ut.demo_top_n_regions(10))

#wr = WeatherReader()
#print(wr.get_vars())
#print(wr.longitudes().size*wr.latitudes().size)
#wr.print_vars_texfmt()

start = datetime(2015,1,8)
stop = datetime(2017,12,31)
fc_end = datetime(2018,12,31)

#for i in range(3,6):
    #armax = ARMAXForecast(start,stop,i,i,['load_lag','t2m_mean'])
    #armax.train()
    #armax.summary()
    #print(armax.predict_one_step_ahead(fc_end))

armax = ARMAXForecast(start,stop,2,2,['load_lag'])
armax.train()
armax.summary()
fc = armax.predict_one_step_ahead(fc_end)
print(fc)

wvars = ['u10', 'v10', 't2m', 'lai_hv', 'lai_lv', 'lcc',
         'stl1', 'slhf', 'str', 'sshf', 'tcc', 'tcrw', 'fdir']

for var in wvars:
    print(var)
    var = f'{var}_mean'
    armax = ARMAXForecast(start,stop,2,2,['load_lag',var])
    armax.train()
    armax.summary()
    fc = armax.predict_one_step_ahead(fc_end)
    print(fc)

#for i in ['t2m_median','t2m_min','t2m_max','t2m_mean']:
    
    #armax = ARMAXForecast(start, stop, 1, 1, exog=[i], const=True)
    #armax.train()
    #armax.summary()
    #fc = armax.predict_one_step_ahead(fc_end)
    #print(fc)


def best_model():
    lr = LoadReader()
    wr = WeatherReader()
    delta1h = timedelta(hours=1)
    start = datetime(2015, 1, 8, 0)
    stop = datetime(2017, 12, 31, 23)
    last_date = datetime(2018, 12, 31, 23)
    
    date_range = pd.date_range(start, last_date, freq='1H')
    fc_range = pd.date_range(stop, last_date-delta1h, freq='1H')
    
    load_data = pd.DataFrame(data={'load': lr.vals4slice(de_load, start, last_date, step=1)},
                             index=date_range)
    
    dow = date_range.to_series().dt.dayofweek.values
    exog = pd.DataFrame(data=[],index=date_range)
    #exog['weekend'] = np.bitwise_or(dow==5,dow==6).astype(int)
    #exog['t2m_mean'] = wr.mean_slice('t2m',start,last_date)
    exog['load_lag'] = lr.vals4slice(de_load,start-timedelta(weeks=1),last_date-timedelta(weeks=1),step=1)
    
    for p in range(1,6):
        for q in range(5):
            armax = ARMA(endog=load_data[start:stop].values,
                         exog =exog[start+delta1h:stop+delta1h].values,
                         order=(p,q))
            armax_result = armax.fit(method='mle',
                                     trend='nc',
                                     #solver='cg',
                                     #transparams=False,
                                     disp=0)
            print(armax_result.summary())
            print(f'ARMA({p},{q})\nAIC:{np.round(armax_result.aic,2)}\nBIC:{np.round(armax_result.bic,2)}\nHQIC:{np.round(armax_result.hqic,2)}')


def plot_corr_t2mmean_load():
    wr = WeatherReader()
    lr = LoadReader()
    
    start = datetime(2015,1,1)
    stop = datetime(2018,12,31)
    
    t2mmean = wr.meanvals4timeslice('t2m',start,stop)
    load_data = lr.vals4slice(de_load, start, stop, step=1)
    
    plt.scatter(t2mmean,load_data,s=1)
    plt.ylabel(de_load)
    plt.xlabel('2 meter temperature')
    plt.show()

#plot_corr_t2mmean_load()

def top10demo_fc():
    # read demo file
    demo_df = pd.read_csv(demography_file,encoding='latin1',index_col='GEO')
    # clean population data
    demo_df['Value'] = demo_df['Value'].map(lambda val: pd.NaT if val == ':' else float(val.replace(',','')))
    # filter by any year, as regions don't actually move, right?
    demo_df = demo_df[demo_df['TIME']==2018]
    # filter all regions with an id of length 5 all others are countries etc
    demo_df = demo_df[[len(reg)==5 for reg in demo_df.index]]
    # sort 
    demo_df.sort_values('Value', axis=0, ascending=False, inplace=True, kind="quicksort", na_position="last")
    
    log.info(demo_df.size)
    wr = WeatherReader()
    lr = LoadReader()
    #allor = wr.check_isinRegion(demo_df.index[0])
    ##allor = np.bitwise_or(allor,wr.check_isinRegion(demo_df.index[1]))
    #for i in range(1,10):
        #reg = wr.check_isinRegion(demo_df.index[i])
        #allor = np.bitwise_or(allor,reg)
    #for i in demo_df.index:
        #if not os.path.exists(os.path.join(isin_path,f'isin{i}.npy')):
            #reg = wr.check_isinRegion(i)
            #if reg is not None:
                #allor = np.bitwise_or(allor,wr.check_isinRegion(i))
    start = datetime(2017,1,1)
    stop = datetime(2017,12,31)
    armax = ARMAXForecast(start, stop, p=1, q=0)
    armax.train()
    armax.summary()
    
    end = stop+timedelta(weeks=1)
    forecast1W = armax.predict_one_step_ahead(end,[1,6,24])
    data = lr.vals4slice(de_load,stop,end,step=1)
    fc_range = pd.date_range(stop,end,freq='1H')
    
    fig,ax = plt.subplots()
    for i,hours in enumerate([1,6,24]):
        ax.plot(fc_range,forecast1W[i],label=f'{hours}H forecast')
        log.info(armax.forecasts[i])
    ax.plot(fc_range,data,label='actual value')
    ax.set_ylabel('load [MW]')
    plt.setp(ax.get_xticklabels(),rotation=30,horizontalalignment='right')
    plt.legend()
    plt.show()

#print(allor)
#plt.imshow(allor)
#plt.show()

#demo_df = pd.read_csv(demography_file,encoding='latin1',index_col='GEO')
#demo_df['Value'] = demo_df['Value'].map(lambda val: pd.NaT if val == ':' else float(val.replace(',','')))
#demo_df = demo_df[demo_df['TIME']==2017]
#demo_df = demo_df[[len(reg)==5 for reg in demo_df.index]]
#demo_df.sort_values('Value', axis=0, ascending=False, inplace=True, kind="quicksort", na_position="last")
#with shp.Reader(nuts3_01res_shape) as nuts3_sf:
    #regions = [rec for rec in nuts3_sf.shapeRecords() if rec.record['CNTR_CODE'] == 'DE']

#def get_region(geo):
    #for reg in regions:
        #if reg.record.NUTS_ID == geo:
            #return reg
        
#def plot_shape(shape_id):
    #region = get_region(shape_id).shape
    #points = np.array(region.points)
    #intervals = list(region.parts) + [len(region.points)]
    #for (x, y) in zip(intervals[:-1], intervals[1:]):
        #plt.plot(*zip(*points[x:y]), color='k', linewidth=2)
    #plt.show()

#plot_shape('DE300')

#print(demo_df)
#print(dir(demo_df))
#print(dir(regions[0].record))
##print(regions[0].record.NUTS_NAME.strip('\000'),regions[0].record.NUTS_ID)

#for i in range(10):
    #print(get_region(demo_df.iloc[i].name).record.NUTS_NAME.strip('\000'))


def data_csv_timet2mload():
    wr = WeatherReader()
    lr = LoadReader()
    
    start = datetime(2015,1,1)
    stop = datetime(2018,12,31)
    
    dr = pd.date_range(start,stop,freq='1H').to_series().to_xarray().values
    tmean = wr.meanvals4timeslice('t2m',start,stop).values
    load = lr.vals4slice(de_load,start,stop,step=1).values
    
    pd.DataFrame({'t2m mean': tmean, 'load' : load},
                 dr).to_csv(os.path.join(data_path,'Dataframe_time_t2mmean_load.csv'),
                            index_label='time')
    #print(xr.Dataset({'time' : dr, 't2m mean' : tmean, 'load' : demand}))#.to_dataframe().to_csv(os.path.join(data_path,'Dataframe_time_t2mmean_load.csv')))


#cat = np.zeros((tr.size,7))
#for i in range(7):
    #cat[tr==i,i] = 1
#print(cat)
#lst = []
#lst.extend(cat)
#print(lst)

#print(np.vstack((tr,tr,tr)))
#ts = pd.Timestamp(2017,1,1)
#print(tr[0], type(tr[0]))
#print(ts,type(ts))
#print(ts.to_pydatetime()+timedelta(hours=2))

#de_holidays = holidays.Germany()
#print(pd.Categorical(tr.dt.dayofweek))

#from statsmodels.tsa.arima_model import ARMA,_make_arma_exog
#pq = (1,0)
#data = np.random.random(100) + np.arange(1,101,1)
##data = lreader.vals4slice(de_load, start, stop).ffill(dim='utc_timestamp').values
#armax_result = ARMA(data,order=pq,exog=np.array(range(len(data)))).fit(trend='c')
#fit_params = armax_result.params
#print(armax_result.k_ar,armax_result.k_trend)
#print(np.array(range(len(data),len(data)+24)))
#print(armax_result.model.exog[-armax_result.k_ar:,armax_result.k_trend:])
#fc = [armax_result.forecast(steps=24,exog=np.array(range(len(data),len(data)+24)))[0][-1]]
##print(armax_result.model.exog.shape,armax_result.k_trend,data.shape)
#for i in range(1,24):
    #data = np.append(data,[np.random.random()+len(data)])
    #armax = ARMA(data,order=pq,exog=np.array(range(len(data))))
    #_,armax.exog = _make_arma_exog(None,np.array(range(len(data))),'c')
    #print(armax_result.k_ar,armax_result.k_trend)
    #print(np.array(range(len(data),len(data)+24)))
    #print(armax_result.model.exog[-armax_result.k_ar:,armax_result.k_trend:])
    #armax.method = 'css-mle'
    #armax_result.initialize(armax,fit_params)
    #fc.append(armax_result.forecast(steps=24,exog=np.array(range(len(data),len(data)+24)))[0][-1])

#print(fc)

#from operator import add,itemgetter
#sums = []
#for (aic,bic,hqic) in zip(aics.items(),bics.items(),hqics.items()):
    #sums.append((aic[0],aic[1]+bic[1]+hqic[1]))

#print(sorted(sums,key=itemgetter(1)))
#print(sorted(hqics.items(),key=itemgetter(1)))

#aics = zip(aics.keys,aics.values)
#bics = zip(bics.keys,bics.values)
#hqics = zip(hqics.keys,hqics.values)

