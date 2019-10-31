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
from datetime import datetime, timedelta, time

def armax_2_2_fc_load_lag():
    """Plot an ARMAX(2,2) forecast, including a 1 week load lag as exogenous variable"""
    plotter = DataPlotter()
    t_start = datetime(2015, 1, 8)
    t_stop = datetime(2017, 12, 31)
    end = datetime(2018, 12, 31)
    
    plot_start = datetime(2018, 1, 1)
    plot_end = datetime(2018, 1, 8)
    plotter.plot_armax_forecast(t_start, t_stop, end, 2, 2, exog=['load_lag168h'], plot_range=(plot_start, plot_end))

def plot_isin_map():
    plotter = DataPlotter()
    plotter.plot_isin()

def plot_t2m_mean():
    """Plot the 2 metre temperature of the day with the highest mean temperature"""
    plotter = DataPlotter()
    plotter.plot_nmax_mean('t2m',1)
