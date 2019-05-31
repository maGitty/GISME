import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type':'reanalysis',
        'variable':[
            '10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature',
            'instantaneous_moisture_flux','low_cloud_cover','surface_net_thermal_radiation',
            'surface_sensible_heat_flux','total_cloud_cover','total_precipitation'
        ],
        'year':[
            '2014','2015','2016',
            '2017','2018','2019'
        ],
        'month':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12'
        ],
        'day':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12',
            '13','14','15',
            '16','17','18',
            '19','20','21',
            '22','23','24',
            '25','26','27',
            '28','29','30',
            '31'
        ],
        'time':[
            '00:00','06:00','12:00',
            '18:00'
        ],     
        'format':'netcdf'
    },
    'download.nc')