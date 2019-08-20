from glob_vars import demography_file,lon_col,lat_col,nuts3_01res_shape,figure_path,de_load
from LoadReader import LoadReader

import os
import numpy as np
import pandas as pd
import shapefile as shp
from datetime import datetime
from descartes import PolygonPatch
from matplotlib import pyplot as plt, cm, colors
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


class Utility:
    """TODO
    
    """
    def __init__(self,save,show,fmt='pdf'):
        """TODO
        
        """
        self.save=save
        self.show=show
        self.fmt=fmt
        with shp.Reader(nuts3_01res_shape) as nuts3_sf:
            self.regions = [rec for rec in nuts3_sf.shapeRecords() if rec.record['CNTR_CODE'] == 'DE']
        
        demo_df = pd.read_csv(demography_file,encoding='latin1',index_col='GEO')
        demo_df['Value'] = demo_df['Value'].map(lambda val: pd.NaT if val == ':' else float(val.replace(',','')))
        self.demo_df = demo_df
    
    def __save_show_fig(self,fig,dir_pth,file_name):
        """TODO
        
        """
        if self.save:
            print(f'saving plot in {file_name}')
            if not os.path.exists(dir_pth):
                os.makedirs(dir_pth)
            if type(self.fmt) is list:
                for f in self.fmt:
                    fig.savefig(f'{file_name}.{f}', bbox_inches='tight', format=f, optimize=True, dpi=150)
            else:
                fig.savefig(f'{file_name}.{self.fmt}', bbox_inches='tight', format=self.fmt, optimize=True, dpi=150)
        if self.show:
            plt.show()
        plt.close(fig)
    
    def demo_for_year(self,year):
        """TODO
        
        """
        df = self.demo_df[self.demo_df['TIME'] == year]
        values = np.array([df.loc[region.record['NUTS_ID'],:]['Value'] for region in self.regions]) / 1000
        _min = values.min()
        _max = values.max()
        
        fig,ax = plt.subplots()
        plt.xlim([5.5,15.5])
        plt.ylim([47,55.5])
        ax.set_xlabel(lon_col)
        ax.set_ylabel(lat_col)
        
        # for logarithmic colorbar
        cbox_bound = np.exp(np.linspace(np.log(_min),np.log(_max),256))
        norm = colors.BoundaryNorm(cbox_bound, ncolors=256)
        sm = cm.ScalarMappable(norm=norm,cmap=cm.get_cmap('jet'))
        cbar = plt.colorbar(sm)
        cbar.set_label('inhabitants (in 1k)')
        
        for value,region in zip(values,self.regions):
            ax.add_patch(PolygonPatch(region.shape.__geo_interface__,fc=sm.to_rgba(value),ec='none'))
        
        dir_pth = os.path.join(figure_path,'demo')
        file_name = os.path.join(dir_pth,f'demo{year}_logscale')
        
        self.__save_show_fig(fig, dir_pth, file_name)
    
    def plot_load_acf(self,start,stop,lags=42,hour_steps=1,ndiff=0):
        """TODO
        
        """
        rd = LoadReader()
        data = rd.vals4step(de_load,step=hour_steps).interpolate_na(dim='utc_timestamp',method='linear').diff(dim='utc_timestamp',n=ndiff).values
        
        fig = plot_acf(data,fft=True,use_vlines=True,lags=lags)
        
        dir_pth = os.path.join(figure_path,'ACF')
        file_name = os.path.join(dir_pth,f'load_{lags}lags_ndiff{ndiff}_hstep{hour_steps}')
        self.__save_show_fig(fig, dir_pth, file_name)
    
    def plot_load_pacf(self,start,stop,lags=42,hour_steps=1,ndiff=0):
        """TODO
        
        """
        rd = LoadReader()
        data = rd.vals4step(de_load,step=hour_steps).interpolate_na(dim='utc_timestamp',method='linear').diff(dim='utc_timestamp',n=ndiff).values
        
        fig = plot_pacf(data,use_vlines=True,lags=lags)
        
        dir_pth = os.path.join(figure_path,'PACF')
        file_name = os.path.join(dir_pth,f'load_{lags}lags_ndiff{ndiff}_hstep{hour_steps}')
        self.__save_show_fig(fig, dir_pth, file_name)
        


ut = Utility(save=True,show=True)

hstep = 2
ut.plot_load_acf(datetime(2015,1,1),datetime(2017,12,31),hour_steps=hstep)


#rd = LoadReader()
#data = rd.vals4step(de_load,step=2).interpolate_na(dim='utc_timestamp',method='linear').diff(dim='utc_timestamp',n=1).values
#plt.plot(data)
#plt.show()

#ut.demo_for_year(2016)

#sf = shp.Reader(nuts3_01res_shape)
#records = [rec for rec in sf.shapeRecords() if rec.record['CNTR_CODE'] == 'DE']

#df = pd.read_csv(demography_file,encoding='latin1',index_col='GEO')

## convert data values from string to floats
#df['Value'] = df['Value'].map(lambda val: pd.NaT if val == ':' else float(val.replace(',','')))
#df = df[df['TIME'] == 2016]

#values = pd.array([df.loc[region.record['NUTS_ID'],:]['Value'] for region in records])
#_min = values.min()
#_max = values.max()


#fix,ax = plt.subplots()
#plt.xlim([5.5,15.5])
#plt.ylim([47,55.5])
#ax.set_xlabel(lon_col)
#ax.set_ylabel(lat_col)

## for logarithmic colorbar, however as values are assigned by hand, they would have to be scaled too
#cbox_bound = np.exp(np.linspace(np.log(_min),np.log(_max),256))
##cbox_bound = np.linspace(_min,_max,256)
#norm = colors.BoundaryNorm(cbox_bound, ncolors=256)
#cm_jet = cm.get_cmap('jet')
#sm = cm.ScalarMappable(norm=norm,cmap=cm_jet)
#cbar = plt.colorbar(sm)
#cbar.set_label('inhabitants')

#for index,region in enumerate(records):
    #value = values[index]
    ##value = np.exp(np.log(values[index]-_min)/np.log(_max))
    ##value = df[(df['GEO'] == record.record['NUTS_ID']) & (df['TIME'] == 2016)]['Value'].iloc[0]/_max
    
    #ax.add_patch(PolygonPatch(region.shape.__geo_interface__,fc=sm.to_rgba(value),ec='none'))
    
#plt.show()
