from glob_vars import data_path, load_path, era5_path, lon_col, lat_col, bbox, de_load
from WeatherReader import WeatherReader
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
from shapely.geometry import Point, Polygon


#rd = NC_Reader()
#t2ms = rd.vals4time('t2m', datetime(2017,1,1,12))[0].flatten()
#tccs = rd.vals4time('tcc', datetime(2017,1,1,12))[0].flatten()
#plt.scatter(tccs,t2ms,s=2)
#plt.show()

#sf = shp.Reader('/home/marcel/Dropbox/data/shapes/DEU_adm1.shp').shapes()
#print(len(sf))

#for shape in sf:
    #points = np.array(shape.points)
    #intervals = list(shape.parts) + [len(shape.points)]
    
    #for (x, y) in zip(intervals[:-1], intervals[1:]):
        #plt.plot(*zip(*points[x:y]), color='k', linewidth=.4)    

#plt.show()
#start = pd.Timestamp(datetime(2016,1,1,0),tz='utc')
#stop = pd.Timestamp(datetime(2017,12,31,18),tz='utc')

##with xr.open_dataset(load_path) as lf:
    ##print(lf[de_load].sel(utc_timestamp=pd.date_range(start,stop,freq='6H')))
    ##print(lf[de_load].sel(utc_timestamp=time(12)))
#ld = LoadReader()
#print(ld.vals4slice(de_load, start, stop))

#with xr.open_mfdataset(f'{era5_path}*.nc') as nc:
    #print(nc['time'].values)


def isinDE():
    # TODO try again on faster pc
    rd = WeatherReader()
    lons = rd.get_coords()[lon_col].values
    lats = rd.get_coords()[lat_col].values
    #coords = [[x,y] for x in lons for y in lats]
    slctr =  [[]]
    sf = shp.Reader('/home/marcel/Dropbox/data/shapes/NUTS_RG_60M_2016_4326_LEVL_0.shp')
    for record in sf.shapeRecords():
        if 'DE' in record.record:
            de_shape = record.shape
    
    poly = Polygon(de_shape.points)
    print(len(lons),len(lats))
    coords = np.empty((len(lats),len(lons)),np.dtype(Point))
    
    print(np.dtype(Point))
    
    
    for y in range(len(lats)):
        for x in range(len(lons)):
            lo = lons[x]
            la = lats[y]
            coords[y,x] = Point(lo,la)

    contains = np.vectorize(lambda p: p.within(poly) or p.touches(poly))
    
    contained = contains(coords)
    print(contained)
    np.save(f'{data_path}isin', contained)
    return contained

#print(len(pd.date_range(datetime(2015,1,1),datetime(2019,3,31),freq='1D')))

#contained = isinDE()

contained = np.load(f'{data_path}isin.npy')

print(np.unique(contained, return_counts=True), contained.size)
#print(type(contained))

#plt.imshow(contained,cmap=plt.cm.gray, extent=bbox)#,interpolation='bilinear')
#plt.show()

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

#plot3dim()

#sf = shp.Reader('/home/marcel/Dropbox/data/shapes/NUTS_RG_60M_2016_4326_LEVL_0.shp/NUTS_RG_60M_2016_4326_LEVL_0.shp')
#print(sf)
#for shape in sf.shapeRecords():
    #if 'DE' in shape.record:
        #print(dir(shape.shape))
        #print(shape.shape.parts, shape.shape.bbox)
        #x = [i[0] for i in shape.shape.points[:]]
        #y = [i[1] for i in shape.shape.points[:]]
        #plt.plot(x,y)
#plt.show()
    
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