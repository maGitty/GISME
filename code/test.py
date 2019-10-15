#!/usr/bin/python3

"""TODO

"""

__author__ = "Marcel Herm"
__credits__ = ["Marcel Herm","Nicole Ludwig","Marian Turowski"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Marcel Herm"
__status__ = "Production"

from glob_vars import (data_path,load_path,era5_path,
                       lon_col,lat_col, bbox,de_load,
                       variable_dictionary,nuts3_01res_shape,
                       nuts0_shape,demography_file,log,isin_path)
from WeatherReader import WeatherReader
from LoadReader import LoadReader
from Predictions import ARMAX_forecast,TSForecast

from netCDF4 import Dataset, num2date
from matplotlib import pyplot as plt
import numpy as np
import geopandas as gpd
from descartes import PolygonPatch
import shapefile as shp
import pandas as pd
from datetime import datetime,timedelta, time
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from glob import glob
import xarray as xr
import pytz
import re,os,operator,functools
from shapely.geometry import Point, Polygon
import holidays
from statsmodels.tsa.arima_model import ARMA


def mape(actual,forecast):
    return np.mean(np.abs((actual-forecast)/actual)*100)

def rmse(actual,forecast):
    return np.sqrt(((forecast-actual)**2).mean())


lr = LoadReader()
wr = WeatherReader()
delta1h = timedelta(hours=1)
start = datetime(2015,1,1,0)
stop = datetime(2017,12,31,23)
last_date = datetime(2018,12,31,23)
#TODO manual forecast
date_range = pd.date_range(start,last_date,freq='1H')
load_data = pd.DataFrame(data={'load' : lr.vals4slice(de_load,start,last_date,step=1)},
                         index=date_range)

dow = date_range.to_series().dt.dayofweek.values
exog = pd.DataFrame(data=[],index=date_range)
exog['weekend'] = np.bitwise_or(dow==5,dow==6).astype(int)
exog['t2m_mean'] = wr.mean_slice('t2m',start,last_date)

start_time = datetime.now()
log.info(f'starting training and forecast: {start_time}')

p=1
q=0
armax = ARMA(endog=load_data[start:stop].values,
             exog =exog[start+delta1h:stop+delta1h].values,
             order=(p,q))
armax_result = armax.fit(method='css',
                         #solver='cg',
                         trend='nc',
                         transparams=False,
                         full_output=-1,
                         disp=0)
fit_params = armax_result.params
log.info(f'fit parameters: {fit_params}\n{armax_result.summary()}')

fc_load = load_data[stop:last_date-delta1h].values[:,0]*fit_params[-1]
fc_exog = np.divide(exog[stop+delta1h:last_date].values.transpose(),fc_load).transpose()
fc_exog = np.dot(fc_exog,fit_params[:-1])
forecast = pd.DataFrame(data={'fc':fc_load+fc_exog if 'fc_exog' in locals() else fc_load},
                       index=pd.date_range(stop,last_date-delta1h,freq='1H'))

print(f'mape: {mape(load_data[stop:last_date-delta1h].values,forecast.values)}')
print(f'rmse: {rmse(load_data[stop:last_date-delta1h].values,forecast.values)}')

fc = TSForecast(1,forecast.values,
                load_data[stop+delta1h:last_date].values,
                start,stop,stop+delta1h,last_date)
print(f'TSForecast: {fc}')

plt.plot(fc.actual,label='actual')
plt.plot(fc.forecast,label='1H forecast')
plt.legend()
plt.show()


def rolling_reestimation_fc():
    lr = LoadReader()
    wr = WeatherReader()
    delta1h = timedelta(hours=1)
    pq=(1,0)
    start = datetime(2015,1,1)
    stop = datetime(2017,12,31)
    last_date = datetime(2018,12,31)    
    load_data = lr.vals4slice(de_load,start,last_date,step=1)
    date_range = pd.date_range(start,last_date,freq='1H')
    load_data = pd.DataFrame(data={'load' : load_data},index=date_range)
    
    dow = date_range.to_series().dt.dayofweek.values
    t2m_mean = wr.mean_slice('t2m',start,last_date)
    exog = pd.DataFrame(data={
                              't2m_mean':t2m_mean,
                              #'weekend':np.bitwise_or(dow==5,dow==6).astype(int)
                              },
                        index=date_range)
    
    start_time = datetime.now()
    log.info(f'starting training and forecast: {start_time}')
    
    armax = ARMA(endog=load_data[start:stop].values,
                 exog =exog[start:stop].values,
                 order=pq)
    
    armax_result = armax.fit(method='css',
                             solver='cg',
                             trend='nc',
                             transparams=True,
                             full_output=-1,
                             disp=0,
                             maxiter=200)
    fit_params = armax_result.params
    
    df = pd.DataFrame(data=np.array([load_data[start:stop].values[:-1][:,0],
                                     armax_result.fittedvalues]).transpose(),
                      columns=['load','fc'],
                      index=pd.date_range(start,stop,freq='1H')[:-1])
    print(f'mape: {mape(df["load"],df["fc"])}')
    print(f'rmse: {rmse(df["load"],df["fc"])}')
    plt.plot(df['load'][:300],label='actual')
    plt.plot(df['fc'][:300],label='fc')
    plt.legend()
    plt.show()
    
    fc = []
    fc.append(armax_result.forecast(steps=24,
                                    exog=exog[stop+delta1h:stop+timedelta(hours=24)]
                                    )[0][-1])
    
    stop_date = stop
    while stop_date < last_date-timedelta(hours=24):
        stop_date += delta1h
        armax = ARMA(endog=load_data[start:stop_date].values,
                     exog =exog[start:stop_date].values,
                     order=pq)
        armax_result = armax.fit(method='css',
                                 solver='cg',
                                 start_params=fit_params,
                                 trend='nc',
                                 transparams=True,
                                 full_output=-1,
                                 disp=0,
                                 maxiter=200)
        fc.append(armax_result.forecast(steps=24,
                                        exog=exog[stop_date+delta1h:stop_date+timedelta(hours=24)]
                                        )[0][-1])
        if stop_date.hour == 23:
            print(f'fc done for {stop_date.date()}')
    log.info(f'finished training and forecast, took {(datetime.now()-start_time).seconds}s')
    plt.plot(load_data[stop+timedelta(hours=24):last_date].index,load_data[stop+timedelta(hours=24):last_date].values,label='actual')
    plt.plot(load_data[stop+timedelta(hours=24):last_date].index,fc,label='24H forecast')
    plt.legend()
    plt.show()

#start = datetime(2017,1,1)
#last_date = datetime(2018,12,31)
#lr = LoadReader()
#wr = WeatherReader()
#date_range = pd.date_range(start,last_date,freq='1H')
#load_data = lr.vals4slice(de_load,start,last_date,step=1).values
##load_data = xr.DataArray(lr.vals4slice(de_load,start,last_date,step=1).values, dims={'time' : date_range})
##load_data = xr.DataArray(load_data,coords=[date_range],dims=['time'])
#data = pd.DataFrame(data={'load' : load_data},index=date_range)
#dow = date_range.to_series().dt.dayofweek.values
##dow = xr.DataArray(np.stack({'load':load_data,'dayofweek':dow}),coords=[date_range],dims=['time'])

##print(dow)
##load_data.merge(xr.Dataset(dow,coords={'time' : date_range}))
#data['Monday'] = (dow == 0).astype(int)
#data['Tuesday'] = (dow == 1).astype(int)
#data['Wednesday'] = (dow == 2).astype(int)
#data['Thursday'] = (dow == 3).astype(int)
#data['Friday'] = (dow == 4).astype(int)
#data['Saturday'] = (dow == 5).astype(int)
#data['Sunday'] = (dow == 6).astype(int)
#print(data[datetime(2017,1,1):datetime(2017,1,2)].values.transpose())
#fdirs = wr.flattened_slice('t2m',start,last_date)

#data.append(pd.DataFrame(fdirs))
#for i, fdir in enumerate(fdirs):
    #data[f'fdir{i}'] = fdir

#print(data[datetime(2017,2,1):datetime(2017,3,1)]['load'].values)
#print(data.values.T.shape)
#print(data)


def plot_corr_t2mmean_load():
    wr = WeatherReader()
    lr = LoadReader()
    
    start = datetime(2015,1,1)
    stop = datetime(2018,12,31)
    
    t2mmean = wr.mean_slice('t2m',start,stop)
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
    armax = ARMAX_forecast(start, stop, p=1, q=0)
    armax.train()
    armax.summary()
    
    end = stop+timedelta(weeks=1)
    forecast1W = armax.predict_range(end,[1,6,24])
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
    tmean = wr.mean_slice('t2m',start,stop).values
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

def containsReg():
    wr = WeatherReader()
    lons = wr.get_coords()[lon_col].values
    lats = wr.get_coords()[lat_col].values
    
    de_shape = shp.Reader(nuts3_01res_shape)
    
    for shape in de_shape.shapeRecords()[30:]:
        if 'DE' in shape.record:
            re_shape = shape
            break
    print(re_shape.record)
    
    poly = Polygon(re_shape.shape.points)
    
    coords = np.empty((len(lats),len(lons)),np.dtype(Point))
    
    for y in range(len(lats)):
        for x in range(len(lons)):
            lo = lons[x]
            la = lats[y]
            coords[y,x] = Point(lo,la)
    
    contains = np.vectorize(lambda p: p.within(poly) or p.touches(poly))
    
    contained = contains(coords)
    plt.imshow(wr.vals4time('t2m',datetime(2017,1,1,12)).where(contained).values)
    plt.show()
    #np.save(f'{data_path}isin', contained)


def plt2d():
    rd = WeatherReader()
    ld = LoadReader()
    
    var = 't2m'
    ystart = 2015
    ystop = 2018
    
    start = pd.Timestamp(datetime(ystart,1,1,0),tz='utc')
    stop = pd.Timestamp(datetime(ystop,12,31,18),tz='utc')
    print(len(pd.date_range(start,stop,freq='1H')))
    
    #print(ld.vals4slice(de_load, start, stop))
    load = ld.vals4slice(de_load, start, stop, step=6)
    #ncval = rd.var_over_time(var).sel(time=slice('2016-1-1','2017-12-31')).values
    ncval = rd.reduce_lonlat(var, np.min).sel(time=slice(f'{ystart}-1-1',f'{ystop}-12-31'))
    #rng = pd.date_range(start,stop,freq='6H')
    
    print(load['utc_timestamp'])
    print(ncval['time'])
    print(load.size,ncval.size)
    
    plt.scatter(ncval.values-273.15,load.values,s=4)
    plt.xlabel(f'{var} min reduce over DE (°C)')
    plt.ylabel('DE_load_actual_entsoe_transparency (MW)')
    #cbar = plt.colorbar()
    #cbar.ax.set_ylabel('DE_load_actual_entsoe_transparency (MW)', rotation=-90, va="bottom")
    
    plt.show()

#plt2d()

def plot3dim():
    rd = WeatherReader()
    ld = LoadReader()
    
    var = 't2m'
    ystart = 2015
    ystop = 2018
    
    start = pd.Timestamp(datetime(ystart,1,1,12),tz='utc')
    stop = pd.Timestamp(datetime(ystop,12,31,12),tz='utc')
    
    load = ld.vals4slice(de_load, start, stop, step=24)
    rng = pd.date_range(start,stop,freq='24H')
    ncval = rd.reduce_lonlat(var, np.min).sel(time=rng)
    
    ncwend = ncval.where((ncval['time.weekday'] == 5) | (ncval['time.weekday'] == 6), drop=True)
    ncweek = ncval.where((ncval['time.weekday'] != 5) & (ncval['time.weekday'] != 6), drop=True)
    
    loadwend = load.where((load['utc_timestamp.weekday'] == 5) | (load['utc_timestamp.weekday'] == 6), drop=True)
    loadweek = load.where((load['utc_timestamp.weekday'] != 5) & (load['utc_timestamp.weekday'] != 6), drop=True)
    
    rngwend = rng.where((rng.weekday == 5) | (rng.weekday == 6), other=pd.NaT).dropna()
    rngweek = rng.where((rng.weekday != 5) & (rng.weekday != 6), other=pd.NaT).dropna()
    
    print(ncwend.sizes, ncweek.sizes)
    print(loadwend.sizes, loadweek.sizes)
    
    print(rngwend.size, rngweek.size)
    
    #print(ncval)
    #print(load.size,ncval.size)
    
    plt.scatter(rngwend, loadwend, s=20, c=ncwend, cmap='jet', marker='>', label='weekend')
    plt.scatter(rngweek, loadweek, s=20, c=ncweek, cmap='jet', marker='<', label='workday')
    plt.ylabel(f'{var} min reduce over DE')
    plt.xlabel('date')
    cbar = plt.colorbar()
    plt.legend()
    cbar.ax.set_ylabel('DE_load_actual_entsoe_transparency (MW)', rotation=-90, va="bottom")
    
    plt.show()
