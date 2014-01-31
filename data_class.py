"""
created on Jan 29, 2014

@author: Nikola Jajcay, based on script by Martin Vejmelka> -- https://github.com/vejmelkam/ndw-climate --
"""

import numpy as np
from netCDF4 import Dataset
from datetime import date


DATA_FOLDER = '../data/'


class DataField:
    """
    Class holds the time series of a geophysical field. The fields for reanalysis data are
    3-dimensional - two spatial and one temporal dimension.
    """
    
    def __init__(self, data = None, lons = None, lats = None, time = None):
        """
        Initializes either an empty data set or with given values.
        """
        self.data = data
        self.lons = lons
        self.lats = lats
        self.time = time
        
        
        
    def load(self, filename = None, variable_name = None, dataset = 'ECA-reanalysis'):
        """
        Loads geophysical data from netCDF file for reanalysis or from text file for station data.
        """
        if dataset == 'ECA-reanalysis':
            d = Dataset(DATA_FOLDER + filename, 'r')
            v = d.variables[variable_name]
            
            self.data = v[:] # masked array - only land data, not ocean/sea
            print("Data saved to structure. Shape of the data is %s and info from netCDF file is %s" % (str(self.data.shape), str(v)))
            self.lons = d.variables['longitude'][:]
            self.lats = d.variables['latitude'][:]
            print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
            self.time = d.variables['time'][:] # days since 1950-01-01 00:00
            self.time += date.toordinal(date(1950, 1, 1))
            print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            
            d.close()     
                    
        if dataset == 'ERA-40':
            d = Dataset(DATA_FOLDER + filename, 'r')
            v = d.variables[variable_name]
            
            self.data = v[:]
            print("Data saved to structure. Shape of the data is %s and info from netCDF file is %s" % (str(self.data.shape), str(v)))
            self.lons = d.variables['longitude'][:]
            self.lats = d.variables['latitude'][:]
            print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
            self.time = d.variables['time'][:] # hours since 1900-01-01 00:00
            self.time = self.time / 24.0 + date.toordinal(date(1900, 1, 1))
            print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            
            d.close()
            
        if dataset == 'NCEP':
            d = Dataset(DATA_FOLDER + filename, 'r')
            v = d.variables[variable_name]
            
            self.data = v[:]
            print("Data saved to structure. Shape of the data is %s and info from netCDF file is %s" % (str(self.data.shape), str(v)))
            self.lons = d.variables['lon'][:]
            self.lats = d.variables['lat'][:]
            self.time = d.variables['time'][:] # hours since 1-01-01 00:00
            self.time = self.time / 24.0 - 1.0
            print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            
            d.close()
            
            
            
    def select_date(self, date_from, date_to):
        """
        Selects the date range - date_from is inclusive, date_to is exclusive. Input is date(year, month, day).
        """
        d_start = date_from.toordinal()
        d_to = date_to.toordinal()
        
        ndx = np.logical_and(self.time >= d_start, self.time < d_to)
        self.time = self.time[ndx] # slice time stamp
        self.data = self.data[ndx, ...] # slice data
        
        
        
    def find_date_ndx(self, d):
        """
        Returns index which corresponds to the date. Returns None is the date is not contained in the data.
        """
        d = date.toordinal()
        pos = np.nonzero(self.time == d)
        if len(pos) == 1:
            return pos[0]
        else:
            return None
            
            
            
    def select_months(self, months):
        """
        Subselects only certain months. Input as a list of months number.
        """
        ndx = filter(lambda i: date.fromordinal(int(self.time[i])).month in months, range(len(self.time)))
        
        self.time = self.time[ndx]
        self.data = self.data[ndx]
        
        
        
    def select_lat_lon(self, lats, lons):
        """
        Selects region in lat/lon. Input is for both [from, to], both are inclusive. If None, the dimension is not modified.
        """
        if lats != None:
            lat_ndx = np.nonzero(np.logical_and(self.lats >= lats[0], self.lats <= lats[1]))[0]
        else:
            lat_ndx = np.arange(len(self.lats))
            
        if lons != None:
            lon_ndx = np.nonzero(np.logical_and(self.lons >= lons[0], self.lons <= lons[1]))[0]
        else:
            lon_ndx = np.arange(len(self.lons))
            
        d = self.data
        d = d[..., lat_ndx, :]
        self.data = d[..., lon_ndx]
        self.lats = self.lats[lat_ndx]
        self.lons = self.lons[lon_ndx]
        
        
        
    def extract_day_month_year(self):
        """
        Extracts the self.time field into three fields containg days, months and years.
        """
        n_days = len(self.time)
        days = np.zeros((n_days,))
        months = np.zeros((n_days,))
        years = np.zeros((n_days,))
        
        for i,d in zip(range(n_days), self.tm):
            dt = date.fromordinal(int(d))
            days[i] = dt.day
            months[i] = dt.month
            years[i] = dt.year
            
        return days, months, years
        
        
    def anomalise(self):
        """
        Removes the seasonal/yearly cycle from the data
        """
        delta = self.time[1] - self.time[0]
        if delta == 1:
            # daily data
            day, mon, _ = self.extract_day_month_year()
            for mi in range(1,13):
                mon_mask = (mon == mi)
                for di in range(1,32):
                    sel = np.logical_and(mon_mask, day == di)
                    if np.sum(sel) == 0:
                        continue
                    avg = np.mean(self.data[sel, ...], axis = 0)
                    self.data[sel, ...] -= avg
        elif abs(delta - 30) < 3.0:
            # monthly data
            for i in range(12):
                avg = np.mean(self.data[i::12, ...], axis = 0)
                self.data[i::12, ...] -= avg
        else:
            raise 'Unknown temporal sampling in the field.'
            
            
            
    def get_seasonality(self):
        """
        Removes the seasonality in both mean and variance and returns the seasonal mean and variance arrays.
        """
        delta = self.time[1] - self.time[0]
        if delta == 1:
            # daily data
            seasonal_mean = np.zeros_like(self.data)
            seasonal_var = np.zeros_like(self.data)
            day, mon, _ = self.extract_day_month_year()
            for mi in range(1,13):
                mon_mask = (mon == mi)
                for di in range(1,32):
                    sel = np.logical_and(mon_mask, day == di)
                    if np.sum(sel) == 0:
                        continue
                    seasonal_mean[sel, ...] = np.mean(self.data[sel, ...], axis = 0)
                    self.data[sel, ...] -= seasonal_mean[sel, ...]
                    seasonal_var[sel, ...] = np.std(self.data[sel, ...], axis = 0)
                    if np.any(seasonal_var[sel, ...] == 0.0):
                        print('**WARNING: some zero standard deviations found for date %d.%d' % (di, mi))
                        seasonal_var[seasonal_var == 0.0] = 1.0
                    self.data[sel, ...] /= seasonal_var[sel, ...]
        elif abs(delta - 30) < 3.0:
            # monthly data
            seasonal_mean = np.zeros([12] + list(self.data.shape[1:]))
            seasonal_var = np.zeros([12] + list(self.data.shape[1:]))
            for i in range(12):
                seasonal_mean[i, ...] = np.mean(self.data[i::12, ...], axis = 0)
                self.data[i::12, ...] -= seasonal_mean[i, ...]
                seasonal_var[i, ...] = np.std(self.data[i::12, ...], axis = 0)
                self.data[i::12, ...] /= seasonal_var[i, ...]
        else:
            raise 'Unknown temporal sampling in the field.'
            
        return seasonal_mean, seasonal_var
                
