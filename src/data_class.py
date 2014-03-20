"""
created on Jan 29, 2014

@author: Nikola Jajcay, based on script by Martin Vejmelka> -- https://github.com/vejmelkam/ndw-climate --
"""

import numpy as np
from scipy.signal import detrend
from netCDF4 import Dataset
from datetime import date, timedelta, datetime
import csv
from os.path import split
from distutils.version import LooseVersion



def nanmean(arr, axis = None):
    """
    Computes the mean along the axis, ignoring NaNs
    """
    a = arr.copy()
    if LooseVersion(np.__version__) >= LooseVersion('1.8.0'): # if numpy version is higher than 1.8, use build-in function
        return np.nanmean(arr, axis = axis)
    else:
        mask = np.isnan(a)
        a[mask] = 0. # replace NaNs to 0.
        total = np.sum(a, axis = axis)
        avg = total / np.sum(~mask, axis = axis)

        return avg
        
        
        
def nanvar(arr, axis = None, ddof = 0):
    """
    Computes the variance along the axis, ignoring NaNs
    """  
    a = arr.copy()      
    if LooseVersion(np.__version__) >= LooseVersion('1.8.0'): # if numpy version is higher than 1.8, use build-in function
        return np.nanvar(arr, axis = axis, ddof = ddof)
    else:
        # compute mean
        mask = np.isnan(a)
        a[mask] = 0. # replace NaNs to 0.
        total = np.sum(a, axis = axis, keepdims = True)
        cnt = np.sum(~mask, axis = axis, keepdims = True)
        avg = total / cnt
        
        # compute squared deviation from mean
        a -= avg
        a[mask] = 0. 
        sqr = np.multiply(a, a)

        # compute variance
        var = np.sum(sqr, axis = axis)
        if var.ndim < cnt.ndim:
            cnt = cnt.squeeze(axis)
        dof = cnt - ddof
        var /= dof
        
        return var
        
        
        
def nanstd(arr, axis = None, ddof = 0):
    """
    Computes the standard deviation along the axis, ignoring Nans
    """
    a = arr.copy()
    if LooseVersion(np.__version__) >= LooseVersion('1.8.0'): # if numpy version is higher than 1.8, use build-in function
        return np.nanstd(arr, axis = axis, ddof = ddof)
    else:
        var = nanvar(a, axis = axis, ddof = ddof)
        std = np.sqrt(var)
        
        return std
        
        

class DataField:
    """
    Class holds the time series of a geophysical field. The fields for reanalysis data are
    3-dimensional - two spatial and one temporal dimension. The fields for station data contains
    temporal dimension and location specification.
    """
    
    def __init__(self, data_folder = '../data/', data = None, lons = None, lats = None, time = None):
        """
        Initializes either an empty data set or with given values.
        """
        
        self.data_folder = data_folder
        self.data = data
        self.lons = lons
        self.lats = lats
        self.time = time
        self.location = None # for station data
        self.missing = None # for station data where could be some missing values
        
        
        
    def load(self, filename = None, variable_name = None, dataset = 'ECA-reanalysis', print_prog = True):
        """
        Loads geophysical data from netCDF file for reanalysis or from text file for station data.
        Now supports following datasets: (dataset - keyword passed to function)
            ECA&D E-OBS gridded dataset reanalysis - 'ECA-reanalysis'
            ECMWF ERA-40 gridded reanalysis - 'ERA-40'
            NCEP/NCAR Reanalysis 1 - 'NCEP'
        """
        
        if dataset == 'ECA-reanalysis':
            d = Dataset(self.data_folder + filename, 'r')
            v = d.variables[variable_name]
            
            self.data = v[:] # masked array - only land data, not ocean/sea
            self.lons = d.variables['longitude'][:]
            self.lats = d.variables['latitude'][:]
            self.time = d.variables['time'][:] # days since 1950-01-01 00:00
            self.time += date.toordinal(date(1950, 1, 1))
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            
            d.close()     
                    
        if dataset == 'ERA-40':
            d = Dataset(self.data_folder + filename, 'r')
            v = d.variables[variable_name]
            
            self.data = v[:]
            self.lons = d.variables['longitude'][:]
            self.lats = d.variables['latitude'][:]
            self.time = d.variables['time'][:] # hours since 1900-01-01 00:00
            self.time = self.time / 24.0 + date.toordinal(date(1900, 1, 1))
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            
            d.close()
            
        if dataset == 'NCEP':
            d = Dataset(self.data_folder + filename, 'r')
            v = d.variables[variable_name]
            
            self.data = v[:]
            self.lons = d.variables['lon'][:]
            self.lats = d.variables['lat'][:]
            self.time = d.variables['time'][:] # hours since 1-01-01 00:00
            self.time = self.time / 24.0 - 1.0
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            
            d.close()
            
            
            
    def load_station_data(self, filename, dataset = 'ECA-station'):
        """
        Loads station data, usually from text file. Uses numpy.loadtxt reader.
        """
        
        if dataset == 'Klem_day':
            raw_data = np.loadtxt(self.data_folder + filename) # first column is continous year and second is actual data
            self.data = np.array(raw_data[:, 1])
            time = []
            
            # use time iterator to go through the dates
            y = int(np.modf(raw_data[0, 0])[1]) 
            if np.modf(raw_data[0, 0])[0] == 0:
                start_date = date(y, 1, 1)
            delta = timedelta(days = 1)
            d = start_date
            while len(time) < raw_data.shape[0]:
                time.append(d.toordinal())
                d += delta
            self.time = np.array(time)
            self.location = 'Praha-Klementinum, Czech Republic'
            print("Station data from %s saved to structure. Shape of the data is %s" % (self.location, str(self.data.shape)))
            print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            
        if dataset == 'ECA-station':
            with open(self.data_folder + filename, 'rb') as f:
                time = []
                data = []
                missing = []
                i = 0 # line-counter
                reader = csv.reader(f)
                for row in reader:
                    i += 1
                    if i == 16: # line with location
                        country = row[0][38:].lower()
                        if row[1][-5] == ' ':
                            station = row[1][1:-13].lower()
                        elif row[1][-6] == ' ':
                            station = row[1][1:-14].lower()
                        self.location = station.title() + ', ' + country.title()
                    if i > 20: # actual data - len(row) = 4 as SOUID, DATE, TG, Q_TG
                        value = float(row[2])
                        year = int(row[1][:4])
                        month = int(row[1][4:6])
                        day = int(row[1][6:])
                        time.append(date(year, month, day).toordinal())
                        if value == -9999.:
                            missing.append(date(year, month, day).toordinal())
                            data.append(np.nan)
                        else:
                            data.append(value/10.)
            self.data = np.array(data)
            self.time = np.array(time)
            self.missing = np.array(missing)
            print("Station data from %s saved to structure. Shape of the data is %s" % (self.location, str(self.data.shape)))
            print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            if self.missing.shape[0] != 0:
                print("** WARNING: There were some missing values! To be precise, %d missing values were found!" % (self.missing.shape[0]))
                        
                                            
                    
    def select_date(self, date_from, date_to):
        """
        Selects the date range - date_from is inclusive, date_to is exclusive. Input is date(year, month, day).
        """
        
        d_start = date_from.toordinal()
        d_to = date_to.toordinal()
        
        ndx = np.logical_and(self.time >= d_start, self.time < d_to)
        self.time = self.time[ndx] # slice time stamp
        self.data = self.data[ndx, ...] # slice data
        if self.missing != None:
            missing_ndx = np.logical_and(self.missing >= d_start, self.missing < d_to)
            self.missing = self.missing[missing_ndx] # slice missing if exists
        
        
        
    def find_date_ndx(self, date):
        """
        Returns index which corresponds to the date. Returns None is the date is not contained in the data.
        """
        
        d = date.toordinal()
        pos = np.nonzero(self.time == d)
        if len(pos) == 1:
            return int(pos[0])
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
        
        if self.lats != None and self.lons != None:
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
        else:
            raise Exception('Slicing data with no spatial dimensions, probably station data.')
            
            
            
    def select_level(self, level):
        """
        Selects the proper level from the data. Input should be integer >= 0.
        """
        
        if self.data.ndim > 3:
            self.data = self.data[:, level, ...]
        else:
            raise Exception('Slicing level in single-level data.')
        
        
        
    def extract_day_month_year(self):
        """
        Extracts the self.time field into three fields containg days, months and years.
        """
        
        n_days = len(self.time)
        days = np.zeros((n_days,), dtype = np.int)
        months = np.zeros((n_days,), dtype = np.int)
        years = np.zeros((n_days,), dtype = np.int)
        
        for i,d in zip(range(n_days), self.time):
            dt = date.fromordinal(int(d))
            days[i] = dt.day
            months[i] = dt.month
            years[i] = dt.year
            
        return days, months, years
        
        
        
    def missing_day_month_year(self):
        """
        Extracts the self.missing field (if exists and is non-empty) into three fields containing days, months and years.
        """
        
        if (self.missing != None) and (self.missing.shape[0] != 0):
            n_days = len(self.missing)
            days = np.zeros((n_days,), dtype = np.int)
            months = np.zeros((n_days,), dtype = np.int)
            years = np.zeros((n_days,), dtype = np.int)
            
            for i,d in zip(range(n_days), self.missing):
                dt = date.fromordinal(int(d))
                days[i] = dt.day
                months[i] = dt.month
                years[i] = dt.year
                
            return days, months, years
            
        else:
            raise Exception('Luckily for you, there is no missing values!')
        
        
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
                    avg = nanmean(self.data[sel, ...], axis = 0)
                    self.data[sel, ...] -= avg
        elif abs(delta - 30) < 3.0:
            # monthly data
            _, mon, _ = self.extract_day_month_year()
            for mi in range(1,13):
                sel = (mon == mi)
                if np.sum(sel) == 0:
                    continue
                avg = nanmean(self.data[sel, ...], axis = 0)
                self.data[sel, ...] -= avg
        else:
            raise Exception('Unknown temporal sampling in the field.')
            
            
            
    def get_seasonality(self, DETREND = False):
        """
        Removes the seasonality in both mean and std (detrending is optional) and 
        returns the seasonal mean and std arrays.
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
                    seasonal_mean[sel, ...] = nanmean(self.data[sel, ...], axis = 0)
                    self.data[sel, ...] -= seasonal_mean[sel, ...]
                    seasonal_var[sel, ...] = nanstd(self.data[sel, ...], axis = 0, ddof = 1)
                    if np.any(seasonal_var[sel, ...] == 0.0):
                        print('**WARNING: some zero standard deviations found for date %d.%d' % (di, mi))
                        seasonal_var[seasonal_var == 0.0] = 1.0
                    self.data[sel, ...] /= seasonal_var[sel, ...]
            if DETREND:
                data_copy = self.data.copy()
                self.data = detrend(self.data, axis = 0)
                trend = data_copy - self.data
            else:
                trend = None
        elif abs(delta - 30) < 3.0:
            # monthly data
            seasonal_mean = np.zeros_like(self.data)
            seasonal_var = np.zeros_like(self.data)
            _, mon, _ = self.extract_day_month_year()
            for mi in range(1,13):
                sel = (mon == mi)
                seasonal_mean[sel, ...] = nanmean(self.data[sel, ...], axis = 0)
                self.data[sel, ...] -= seasonal_mean[sel, ...]
                seasonal_var[sel, ...] = nanstd(self.data[sel, ...], axis = 0, ddof = 1)
                self.data[sel, ...] /= seasonal_var[sel, ...]
            if DETREND:
                data_copy = self.data.copy()
                self.data = detrend(self.data, axis = 0)
                trend = data_copy - self.data
            else:
                trend = None
        else:
            raise Exception('Unknown temporal sampling in the field.')
            
        return seasonal_mean, seasonal_var, trend
        
        
        
def load_station_data(filename, start_date, end_date, anom, dataset = 'ECA-station'):
    """
    Data loader for station data.
    """
    
    print("[%s] Loading station data..." % (str(datetime.now())))
    g = DataField()
    g.load_station_data(filename, dataset)
    print("** loaded")
    g.select_date(start_date, end_date)
    if anom:
        print("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()
    print("[%s] Data from %s loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), g.location, str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))
           
    return g
    
    
    
def load_NCEP_data_daily(filename, varname, start_date, end_date, lats, lons, level, anom):
    """
    Data loader for daily reanalyses data. Filename in form path/air.sig995.%d.nc
    """
    
    print("[%s] Loading daily NCEP/NCAR data..." % str(datetime.now()))
    start_year = start_date.year
    end_year = end_date.year - 1
    glist = []
    Ndays = 0
    path, name = split(filename)
    path += "/"
    
    for year in range(start_year, end_year+1):
        g = DataField(data_folder = path)
        fname = name % year
        g.load(fname, varname, dataset = 'NCEP', print_prog = False)
        Ndays += len(g.time)
        glist.append(g)
        
    data = np.zeros((Ndays, len(glist[0].lats), len(glist[0].lons)))
    time = np.zeros((Ndays,))
    n = 0
    for g in glist:
        Ndays_i = len(g.time)
        data[n:Ndays_i + n, ...] = g.data
        time[n:Ndays_i + n] = g.time
        n += Ndays_i
        
    g = DataField(data = data, lons = glist[0].lons, lats = glist[0].lats, time = time)
    del glist
    print("** loaded")
    g.select_date(start_date, end_date)
    g.select_lat_lon(lats, lons)
    if level != None:
        g.select_level(level)
    if anom:
        print("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()

    print("[%s] NCEP data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))
           
    return g
