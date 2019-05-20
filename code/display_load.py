from matplotlib import pyplot, dates
import csv
import pandas
import numpy

power_df = pandas.read_csv("/home/marcel/Projects/GISME/data/power_load/time_series_15min_singleindex_filtered.csv", low_memory=False)
dates = power_df.where(power_df.loc[:, u'DE_load_entsoe_transparency'].notna()).loc[:, u'utc_timestamp'].astype('datetime64[ns]')
data = power_df.where(power_df.loc[:, u'DE_load_entsoe_transparency'].notna()).loc[:, u'DE_load_entsoe_transparency']
pyplot.plot_date(dates, data, '-', linewidth=1)
pyplot.xlabel("date")
pyplot.ylabel("load")
pyplot.show()

