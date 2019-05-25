from netCDF4 import Dataset
from glob_vars import data_path
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import numpy as np

#nc_pth = data_path + 'ecmwf/netcdf_actuals/GridOneDayAhead_2018-01-01.nc'


#nc_file = Dataset(nc_pth, 'r+')

#print(nc_file.variables)
#print(nc_file.dimensions)

#nc_file.close()

m = Basemap(lon_0=10, lat_1=47, lat_2=56, lat_0=51, projection='lcc', resolution='c', width=800000,height=900000)
# meridians on bottom and left
parallels = np.arange(40,61,2)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(0,15,2)
m.drawmeridians(meridians,labels=[True,False,False,True])
m.drawcountries()
m.drawcoastlines()
plt.show()