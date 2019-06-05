from glob_vars import data_path, load_path, era5_path
from NC_Reader import NC_Reader
from LoadReader import LoadReader

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


#nc_pth = f'{data_path}ecmwf/netcdf_actuals/GridOneDayAhead_2018-06-01.nc'

#df = pd.read_pickle(load_path.replace('.csv', '.pkl'))
#print(df.head())
#ts1 = df['cet_cest_timestamp'][0]
#print(ts1, type(ts1))

#ds = xr.open_dataset('/home/marcel/download.nc')
#print(ds['t2m'].sel(time=time(12)).values.ndim)
#print(ds['t2m'].units)

rd = NC_Reader()
ld = LoadReader(filetype='pkl')

var = 't2m'
ystart = 2016
ystop = 2017

start = pd.Timestamp(datetime(ystart,1,1,0),tz='utc')
stop = pd.Timestamp(datetime(ystop,12,31,18),tz='utc')

load = ld.from_range(start, stop, step=True)['DE_load_actual_entsoe_transparency'].get_values()
#ncval = rd.var_over_time(var).sel(time=slice('2016-1-1','2017-12-31')).values
ncval = rd._func_over_time(var, np.min).sel(time=slice(f'{ystart}-1-1',f'{ystop}-12-31'))
rng = pd.date_range(start,stop,freq='6H')

print(load.size,ncval.size)

plt.scatter(rng, ncval,s=4,c=load,cmap='jet')
plt.ylabel(f'{var} min reduce over DE')
plt.xlabel('date')
cbar = plt.colorbar()
cbar.ax.set_ylabel('DE_load_actual_entsoe_transparency (MW)', rotation=-90, va="bottom")

plt.show()


def mfdstest():
    lst = list(map(lambda n: f'{era5_path}{n}.nc',
                   ['ERA5_RSL_2015H1', 'ERA5_RSL_2015H2', 'ERA5_RSL_2016H1', 'ERA5_RSL_2016H2', 'ERA5_RSL_2017H1',
                    'ERA5_RSL_2017H2', 'ERA5_RSL_2018H1', 'ERA5_RSL_2018H2', 'ERA5_RSL_2019H1']
                   ))
    
    s2 = 0
    with xr.open_mfdataset(f'{era5_path}*.nc', decode_times=True) as mf:
        #print([v for v in mf.variables])
        #print(mf.to_dataframe().tail(4))
        #print(mf['t2m'].sel(time='2018-6-8 12', longitude=10, latitude=50).values - 273.15)
        #mi = mf['t2m'].min(dim=['longitude','latitude']).values
        #print(mi)
        #print(mf.where(mf['t2m'] == mf['t2m'].min(), drop=True)['time'].values)
        dr = mf['t2m'].reduce(np.min, dim=['longitude','latitude']).dropna(dim='time')
        #print(dr.sortby(dr)[:4])
        #print(dr.sortby(dr))
        #print(dr.sortby(dr)[-4:])
        dr = mf['t2m'].reduce(np.var, dim=['longitude','latitude'])
        #print(dr)
        print(pd.to_datetime(dr.sortby(dr)['time'].values[0]).strftime("%Y%m%d%H"))
        dr = mf['t2m'].var(dim=['longitude','latitude'])
        #print(dr)
        print(dr.sortby(dr))      
        #print(mf['t2m'].var(dim=['longitude','latitude']).max().values)
        
        #idx = mf.sel(time=time(12))['t2m'].values.reshape(mf.sel(time=time(12))['t2m'].values.size)
        #idx = idx[np.logical_not(np.isnan(idx))]
        #idx.sort()
        #st = set()
        #i = 0
        #while len(st)<3:
            #day = mf.where(mf['t2m'] == idx[i], drop=True)['time'].values
            #st.add(day[0])
            #print(day,idx[i])
            #i+=1
        #print(st)
        
        print(mf.where(mf['t2m'].var(dim=['longitude','latitude']) == mf['t2m'].var(dim=['longitude','latitude']).max(), drop=True)['time'].values)
        print(mf['t2m'].sel(time='2017-01-07 06').var().values)
        print(np.isnan(mf['t2m'].values).any())
        
        vs = mf['t2m'].var(dim=['longitude','latitude'],skipna=True).values
        vs.sort()
        print(vs[np.logical_not(np.isnan(vs))])
    
    
    #print(mf['t2m'].sel(time='2019/01').values.size)
    #print(mf.values())

#ncf1 = xr.open_dataset(f'{era5_path}ERA5_RSL_2015H1.nc')
#print(ncf1['time'].size)

def mergefiles():
    ex_pth = f'{era5_path}ERA5_RSL_2015H1.nc'
    ex1_pth = f'{era5_path}ERA5_RSL_2015H2.nc'
    
    lst = glob(f'{era5_path}*.nc')
    big_nc = xr.open_dataset(lst.pop(), decode_times=True)
    
    while True:
        if not lst:
            print('list is empty')
            break
        small_nc = xr.open_dataset(lst.pop(), decode_times=True)
        big_nc = xr.merge([big_nc, small_nc])
    
    print('hello')
    big_nc.to_netcdf(path='/home/marcel/weather.nc', mode='w')
    print(big_nc)
    
#print(xr.merge(map(lambda x: xr.open_dataset(x, decode_times=True), glob(f'{era5_path}*.nc'))))

#ncf = xr.open_dataset(ex_pth, decode_times=True)
#ncf1 = xr.open_dataset(ex1_pth, decode_times=True)
#print(xr.merge([ncf,ncf1]))

def xpl():
    ex_pth = glob(f'{era5_path}*.nc')[0]
    ex1_pth = glob(f'{era5_path}*.nc')[1]
    
    ncf = xr.open_dataset(ex_pth, decode_times=True)
    #tm = ncf['time']
    
    # TODO like this change time to pseudo timezone, but contains no information about
    # timezone, otherwise xarray wouldn't recognize datetime64 type
    # take first and last date to get a range and convert to local time, then remove timezone information
    aware = pd.date_range(ncf['time'].values[0], ncf['time'].values[-1], freq='6H', tz='UTC').tz_convert('Europe/Berlin').tz_localize(None)
    print(aware)
    ncf['time'] = aware
    print(ncf['time'])
    #print(ncf)
    #print(ncf.variables['time'].values[0] + np.timedelta64(1,'W'))
    #print(pytz.timezone('Europe/Berlin'))

def plottest():
    nc_pth = f'{data_path}../../download.nc'
    nc_file = Dataset(nc_pth, 'r')
    print(nc_file.variables['t2m'].shape)
    print(num2date(nc_file.variables['time'][742], nc_file.variables['time'].units))
    ind = nc_file.variables['latitude'][:]
    cols = nc_file.variables['longitude'][:]
    
    print(nc_file.variables['t2m'][:].count())
    print(cols.min(), cols.max(), ind.min(), ind.max())
    
    df = pd.DataFrame(nc_file.variables['t2m'][:][742], index=ind, columns=cols).apply(lambda x: x-273.15)
    
    plt.imshow(df, cmap='jet', extent=(cols.min(), cols.max(), ind.min(), ind.max()), interpolation='bilinear')
    
    plt.colorbar()
    plt.show()
    
    nc_file.close()    

#times = nc_file.variables['time']
#h = nc_file.variables['t2m']
#print(h[:,0])
#jd = num2date(times[:], times.units)[2]

#hs = pd.Series(map(lambda x: x[0], h[:,0]), index=jd)
#print(hs)

#num = int(nc_file.variables['time'][0])

#print(datetime(1900,1,1) + timedelta(hours=num), nc_file.variables['time'][:])
#print(num2date(num, nc_file.variables['time'].units))

#base = datetime(1900,1,1)

#print([str(x) for x in map(lambda x: timedelta(hours=int(x))+base, nc_file.variables['time'][:])])
#names = [vari for vari in nc_file.variables]

#for name in names:
    #print("name: " + nc_file.variables[name].long_name + "    unit: " + nc_file.variables[name].units)
#print(nc_file.dimensions)

#nc_file.close()

#fig = plt.figure()
#ax = fig.subplots()

#ger_file = gpd.read_file(data_path + 'bundeslaender_simplify20.geojson')
#print(type(ger_file))
#poly_patch = PolygonPatch(ger_file.geometry, ec='black', alpha=1)
#ax.add_patch(poly_patch)


#print(ger_file.total_bounds)
#ax = ger_file.plot()
#plt.show()

#sf = shp.Reader('/home/marcel/Downloads/vg2500_geo84/vg2500_bld.shp')
#for shape in sf.shapeRecords():
    #x = [i[0] for i in shape.shape.points[:]]
    #y = [i[1] for i in shape.shape.points[:]]
    #print(x,y)

#sf = shp.Reader("/home/marcel/Downloads/vg2500_geo84/vg2500_bld.shp")

#print("Initializing Display")
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.xlim([5.7, 15.2])
#plt.ylim([47, 55.4])
#print("Display Initialized")

#for shape in sf.shapes():
    #print("Finding Points")
    #points = shape.points
    #print("Found Points")    

    #print("Creating Polygon")
    #ap = plt.Polygon(points, fill=False, edgecolor="k")
    #ax.add_patch(ap)
    #print("Polygon Created")

#print("Displaying Polygons")
#plt.show()

#sf=shp.Reader('/home/marcel/Downloads/vg2500_geo84/vg2500_bld.shp')
##sf.plot(linestyle='-', rasterized=True)
##plt.show()

#for shape in sf:
    #print(shape)
    #x = [i[0] for i in shape.shape.points[:]]
    #y = [i[1] for i in shape.shape.points[:]]
    #plt.plot(x,y)
#plt.show()


#for i in range(0,4):
    #sf = gpd.read_file(f'/home/marcel/Downloads/DEU_adm/DEU_adm{i}.shp')
    #sf.plot()
    #plt.show()
    
#hertz_load = 'DE_50hertz_load_actual_entsoe_transparency'
#amprion_load = 'DE_amprion_load_actual_entsoe_transparency'
#tennet_load = 'DE_tennet_load_actual_entsoe_transparency'
#transnetbw_load = 'DE_transnetbw_load_actual_entsoe_transparency'

#power_df = pd.read_csv(data_path + "power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
#print(power_df[power_df[hertz_load] == 0])
#print(power_df[power_df[amprion_load] == 0])
#print(power_df[power_df[tennet_load] == 0])
#print(power_df[power_df[transnetbw_load] == 0])

#print(datetime(2017, 1, 1, 12))

#def days_of_month(y, m):
    #d0 = datetime(y, m, 1)
    #d1 = d0 + relativedelta(months=1)
    #out = list()
    #while d0 < d1:
        #out.append(d0.strftime('%Y-%m-%d'))
        #d0 += timedelta(days=1)
    #return out

#years = range(2017,2018)
#months = range(1,13)
#dates = [datetime(year,month,day,12) for year in years for month in months for day in range(1,monthrange(year, month)[1]+1)]

#wvars = []

#weather_df = pd.read_csv(f'{data_path}ecmwf/GridActuals_2017.csv', low_memory=False)
#for date in dates:
    #wvar = weather_df[weather_df['time'] == str(date)].loc[:, 't2m'].var()
    ##print(wvar)
    #wvars.append((date,wvar))

#wvars = sorted(wvars, key=lambda x: -x[1])
#from plot_weather_map import plot_map_matplotlib_csv
#for i in range(5):
    #date = wvars[i][0]
    #plot_map_matplotlib_csv(date)
    
    
#xv = np.linspace(6,15,(15-6)*4+1)
#yv = np.linspace(47.5,55,(55-47.5)*4+1)
#a = np.array(np.meshgrid(xv,yv)).T.reshape(-1,2)

#for i in a:
    #print(i)