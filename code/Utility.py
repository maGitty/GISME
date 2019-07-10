from glob_vars import demography_file,lon_col,lat_col,nuts3_01res_shape,figure_path

import os
import numpy as np
import pandas as pd
import shapefile as shp
from descartes import PolygonPatch
from matplotlib import pyplot as plt, cm, colors


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
    
    def demo_for_year(self,year):
        """TODO
        
        """
        df = self.demo_df[self.demo_df['TIME'] == year]
        values = pd.array([df.loc[region.record['NUTS_ID'],:]['Value'] for region in self.regions])
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
        cbar.set_label('inhabitants')
        
        for value,region in zip(values,self.regions):
            ax.add_patch(PolygonPatch(region.shape.__geo_interface__,fc=sm.to_rgba(value),ec='none'))
            
        if self.save:
            dir_pth = f'{figure_path}demo/'
            file_name = f'{dir_pth}demo{year}_logscale'
            print(f'saving plot for {year} in {file_name}')
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

ut = Utility(save=True,show=True)
ut.demo_for_year(2018)

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
