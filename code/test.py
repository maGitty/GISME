from netCDF4 import Dataset
from glob_vars import data_path
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import numpy as np
import geopandas as gpd
from descartes import PolygonPatch
import shapefile as shp
import pandas as pd
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange


#nc_pth = data_path + 'ecmwf/netcdf_actuals/GridOneDayAhead_2018-01-01.nc'


#nc_file = Dataset(nc_pth, 'r+')

#print(nc_file.variables)
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

def days_of_month(y, m):
    d0 = datetime(y, m, 1)
    d1 = d0 + relativedelta(months=1)
    out = list()
    while d0 < d1:
        out.append(d0.strftime('%Y-%m-%d'))
        d0 += timedelta(days=1)
    return out

years = range(2017,2018)
months = range(1,13)
dates = [datetime(year,month,day,12) for year in years for month in months for day in range(1,monthrange(year, month)[1]+1)]

wvars = []

weather_df = pd.read_csv(f'{data_path}ecmwf/GridActuals_2017.csv', low_memory=False)
for date in dates:
    wvar = weather_df[weather_df['time'] == str(date)].loc[:, 't2m'].var()
    #print(wvar)
    wvars.append((date,wvar))

wvars = sorted(wvars, key=lambda x: -x[1])
from plot_weather_map import plot_map_matplotlib_csv
for i in range(5):
    date = wvars[i][0]
    plot_map_matplotlib_csv(date)