from glob_vars import data_path, load_path, era5_path, lon_col, lat_col, bbox, de_load,variable_dictionary,nuts3_01res_shape
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
import re,os,operator,functools
from shapely.geometry import Point, Polygon
import holidays

tr = pd.date_range(datetime(2017,1,1), datetime(2018,1,1),freq='1H').to_series().dt.dayofweek.values
print(np.vstack((tr,tr,tr)))
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

#aics = {(0, 0): 186714.3548858944, (0, 1): 175633.37537402584, (1, 0): 163027.38845677022, (1, 1): 157176.67343019284, (1, 2): 155342.28241170605, (1, 3): 154716.0389058001, (1, 4): 154582.13010879586, (1, 5): 154263.35948238504, (2, 0): 154249.55496882004, (2, 1): 154069.85174305786, (2, 2): 154058.3584256741, (2, 3): 154058.3887386597, (2, 4): 154043.67231602847, (2, 5): 153902.96073415436, (3, 0): 154095.59900161356, (3, 1): 154062.04068381674, (3, 2): 154059.77340719334, (3, 3): 153861.38907323967, (3, 4): 153598.73786407895, (3, 5): 153459.86752778187, (4, 0): 154052.4711471144, (4, 1): 154054.35603108935, (4, 2): 153596.55464831754, (4, 3): 153404.0479566709, (4, 4): 153404.70392228127, (4, 5): 152872.97157628342, (5, 0): 154054.13821217592, (5, 1): 153847.73179585653, (5, 2): 153815.10269696827, (5, 3): 153484.17443698866, (5, 4): 152980.68518415198, (5, 5): 153404.44066458987, (6, 0): 154012.45594892552, (6, 1): 153815.12878775626, (6, 2): 153692.1066107311, (6, 3): 153055.66460441807, (6, 4): 153438.93510605983, (6, 5): 153225.90746427735, (7, 0): 153943.14267245735, (7, 1): 153933.04093311363, (7, 2): 153929.92135428792, (7, 3): 153606.2687819614, (7, 4): 153316.87507828407, (7, 5): 152080.54380990018}
#bics = {(0, 0): 186728.51101655973, (0, 1): 175654.60957002384, (1, 0): 163048.62265276822, (1, 1): 157204.9856915235, (1, 2): 155377.67273836938, (1, 3): 154758.50729779608, (1, 4): 154631.67656612452, (1, 5): 154319.98400504637, (2, 0): 154277.8672301507, (2, 1): 154105.2420697212, (2, 2): 154100.82681767008, (2, 3): 154107.93519598836, (2, 4): 154100.2968386898, (2, 5): 153966.66332214835, (3, 0): 154130.9893282769, (3, 1): 154104.50907581273, (3, 2): 154109.319864522, (3, 3): 153918.013595901, (3, 4): 153662.44045207294, (3, 5): 153530.64818110853, (4, 0): 154094.9395391104, (4, 1): 154103.902488418, (4, 2): 153653.17917097887, (4, 3): 153467.7505446649, (4, 4): 153475.48457560793, (4, 5): 152950.83029494275, (5, 0): 154103.68466950458, (5, 1): 153904.35631851785, (5, 2): 153878.80528496226, (5, 3): 153554.9550903153, (5, 4): 153058.5439028113, (5, 5): 153489.37744858186, (6, 0): 154069.08047158684, (6, 1): 153878.83137575025, (6, 2): 153762.88726405776, (6, 3): 153133.5233230774, (6, 4): 153523.8718900518, (6, 5): 153317.922313602, (7, 0): 154006.84526045134, (7, 1): 154003.8215864403, (7, 2): 154007.78007294724, (7, 3): 153691.2055659534, (7, 4): 153408.88992760872, (7, 5): 152179.6367245575}
#hqics = {(0, 0): 186719.17833029767, (0, 1): 175640.61054063076, (1, 0): 163034.62362337514, (1, 1): 157186.3203189994, (1, 2): 155354.34102271427, (1, 3): 154730.50923900993, (1, 4): 154599.01216420735, (1, 5): 154282.6532599982, (2, 0): 154259.2018576266, (2, 1): 154081.9103540661, (2, 2): 154072.82875888393, (2, 3): 154075.2707940712, (2, 4): 154062.96609364162, (2, 5): 153924.66623396915, (3, 0): 154107.65761262178, (3, 1): 154076.51101702658, (3, 2): 154076.65546260483, (3, 3): 153880.6828508528, (3, 4): 153620.44336389375, (3, 5): 153483.98474979828, (4, 0): 154066.94148032425, (4, 1): 154071.23808650085, (4, 2): 153615.8484259307, (4, 3): 153425.7534564857, (4, 4): 153428.8211442977, (4, 5): 152899.5005205015, (5, 0): 154071.02026758742, (5, 1): 153867.02557346967, (5, 2): 153836.80819678307, (5, 3): 153508.29165900507, (5, 4): 153007.21412837005, (5, 5): 153433.3813310096, (6, 0): 154031.74972653866, (6, 1): 153836.83428757105, (6, 2): 153716.22383274752, (6, 3): 153082.19354863613, (6, 4): 153467.87577247954, (6, 5): 153257.2598528987, (7, 0): 153964.84817227215, (7, 1): 153957.15815513005, (7, 2): 153956.45029850598, (7, 3): 153635.20944838112, (7, 4): 153348.2274669054, (7, 5): 152114.30792072316}

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


# contained = np.load(f'{data_path}isin.npy')
#
# wr = WeatherReader()
# plt.imshow(wr.vals4time('t2m', datetime(2017,1,1,12)).where(contained).values)
# plt.show()

#sf = shp.Reader('/home/marcel/Dropbox/data/shapes/NUTS_RG_60M_2016_4326_LEVL_3.shp/NUTS_RG_60M_2016_4326_LEVL_3.shp')
#print(sf.fields)
#print(sf.shapeRecords()[52].record['NUTS_NAME'].strip('\000'))

#df = pd.read_csv('/home/marcel/Dropbox/data/demo_r_d3dens/demo_r_d3dens_1_Data.csv',encoding='latin1')
#print(df.columns)
#print(df[df['GEO'] == sf.shapeRecords()[52].record['NUTS_NAME'].strip('\000')]['Value'])

#records = [rec for rec in sf.shapeRecords() if rec.record['CNTR_CODE'] == 'DE']

#for record in records:
    #shape = record.shape
    #points = np.array(shape.points)
    #intervals = list(shape.parts) + [len(shape.points)]
    #for (x, y) in zip(intervals[:-1], intervals[1:]):
        #plt.plot(*zip(*points[x:y]), color='k', linewidth=.4) 
#plt.show()


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

#contained = np.load(f'{data_path}isin.npy')

#print(np.unique(contained, return_counts=True), contained.size)
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