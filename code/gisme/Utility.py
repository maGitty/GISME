#!/usr/bin/python3

"""
This module provides utility functions
"""

from gisme import (demography_file, nuts3_01res_shape, log, nuts0_shape,
                   isin_path, lon_min, lon_max, lat_min, lat_max)
from gisme.LoadReader import LoadReader

import os
import numpy as np
import pandas as pd
import shapefile as shp
from shapely.geometry import Polygon, Point


class Utility:
    """Provides utility functions
    
    Attributes
    ----------
    TODO
    """
    def __init__(self):
        """Initialize Utility instance"""
        self.lats = np.arange(lat_max, lat_min-.1, -.25)
        self.lons = np.arange(lon_min, lon_max+.1, .25)

    @staticmethod
    def get_region(region_id):
        """Return region for a given region_id
        
        Parameters
        ----------
        region_id : string
                    id of region
        
        Returns
        -------
        shapefile.ShapeRecord for region
        """
        # load shapes and mapping of ids to region names
        with shp.Reader(nuts3_01res_shape) as nuts3_sf:
            regions = [rec for rec in nuts3_sf.shapeRecords() if rec.record['CNTR_CODE'] == 'DE']
        # try to find desired region
        for region in regions:
            if region.record.NUTS_ID == region_id:
                # convert region shape to polygon for plotting
                return region
    
    def get_region_name(self,region_id):
        """Return region name for a given region_id
        
        Parameters
        ----------
        region_id : string
                    id of region
        
        Returns
        -------
        name of the region a string
        """
        return self.get_region(region_id).record['NUTS_NAME'].strip('\000')
    
    def demo_top_n_regions(self, n, year):
        """Searches for n most populated regions
        
        Parameters
        ----------
        n    : integer
               specifies number of most populated regions to return
        year : integer
               the year for which to check population
        
        Returns
        -------
        dictionary of most populated regions' names with respective population
        """
        # read demo file
        demo_df = pd.read_csv(demography_file, encoding='latin1', index_col='GEO')
        # clean population data
        demo_df['Value'] = demo_df['Value'].map(lambda val: pd.NaT if val == ':' else float(val.replace(',', '')))
        # filter by any year, as regions don't actually move, right?
        demo_df = demo_df[demo_df['TIME'] == year]
        # filter all regions with an id of length 5 all others are countries etc
        demo_df = demo_df[[len(reg) == 5 for reg in demo_df.index]]
        # sort by population
        demo_df.sort_values('Value', axis=0, ascending=False, inplace=True, kind="quicksort", na_position="last")
        return {self.get_region(region_id).record.NUTS_NAME.strip('\000'): region['Value']
                for region_id, region in demo_df.head(n).iterrows()}

    @staticmethod
    def demo_top_n_regions_map(n, year):
        """Returns 2D array showing which grid points are within the top n regions regarding population
        
        Parameters
        ----------
        n : integer
            specifies number of most populated regions to return
        year : integer
               the year for which to check population
        
        Returns
        -------
        2D numpy.ndarray
        """
        # read demo file
        demo_df = pd.read_csv(demography_file, encoding='latin1', index_col='GEO')
        # clean population data
        demo_df['Value'] = demo_df['Value'].map(lambda val: pd.NaT if val == ':' else float(val.replace(',', '')))
        # filter by any year, as regions don't actually move, right?
        demo_df = demo_df[demo_df['TIME'] == year]
        # filter all regions with an id of length 5 all others are countries etc
        demo_df = demo_df[[len(reg) == 5 for reg in demo_df.index]]
        # sort 
        demo_df.sort_values('Value', axis=0, ascending=False, inplace=True, kind="quicksort", na_position="last")
        
        top10map = np.load(os.path.join(isin_path, f'isin{demo_df.index[0]}.npy'))
        for i in range(1, n):
            reg = np.load(os.path.join(isin_path, f'isin{demo_df.index[i]}.npy'))
            top10map = np.bitwise_or(top10map, reg)
        return top10map
    
    def check_isinDE(self):
        """Used to check which points of dataset are within germany
        
        Returns
        -------
        2D numpy.ndarray containing booleans wether point lies within germany or not
        """
        eu_shape = shp.Reader(nuts0_shape)
        de_shape = None
        for record in eu_shape.shapeRecords():
            if 'DE' in record.record:
                de_shape = record
                break
        if de_shape is None:
            raise Exception('shape for germany could not be found!')
        poly = Polygon(de_shape.shape.points)
        
        coords = np.empty((len(self.lats), len(self.lons)), np.dtype(Point))
    
        for y in range(len(self.lats)):
            for x in range(len(self.lons)):
                lo = self.lons[x]
                la = self.lats[y]
                coords[y, x] = Point(lo, la)
    
        contains = np.vectorize(lambda p: p.within(poly) or p.touches(poly))
    
        contained = contains(coords)
        np.save(os.path.join(isin_path, 'isinDE'), contained)
        
        return contained
    
    def check_isin_region(self, region_id):
        """Computes which points are within specified region
        
        Parameters
        ----------
        region_id : string
                    id of a region
        
        Returns
        -------
        2D numpy.ndarray containing booleans specifying whether point lies within region or not
        """
        region_poly = None
        # try to find desired region
        region_poly = Polygon(self.get_region(region_id).shape.points)
        if region_poly is None:
            raise Exception(f'shape for region {region_id} could not be found!')
        # create 2D numpy.ndarray of points to ease vectorized computation
        coords = np.empty((len(self.lats), len(self.lons)), np.dtype(Point))
        for y in range(len(self.lats)):
            for x in range(len(self.lons)):
                lo = self.lons[x]
                la = self.lats[y]
                coords[y, x] = Point(lo, la)
        # checks for each point whether it is within the polygon of the desired region
        contains = np.vectorize(lambda point: point.within(region_poly) or point.touches(region_poly))
        contained = contains(coords)
        if not os.path.exists(isin_path):
            os.mkdir(isin_path)
        filename = os.path.join(isin_path, f'isin{region_id}')
        log.info(f'isin saved as {filename}')
        np.save(filename, contained)
    
        return contained


#ut = Utility()
#print(ut.demo_top_n_regions(10))
