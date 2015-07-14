"""
created on Jan 29, 2014

@author: Nikola Jajcay, jajcay(at)cs.cas.cz
based on script by Martin Vejmelka
-- https://github.com/vejmelkam/ndw-climate --
"""

import numpy as np
from netCDF4 import Dataset
from datetime import date, timedelta, datetime
import csv
from os.path import split
import os
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
        
        
        
def nandetrend(arr, axis = 0):
    """
    Removes the linear trend along the axis, ignoring Nans.
    """
    a = arr.copy()
    rnk = len(a.shape)
    # determine axis
    if axis < 0:
        axis += rnk # axis -1 means along last dimension
    
    # reshape that axis is 1. dimension and other dimensions are enrolled into 2. dimensions
    newdims = np.r_[axis, 0:axis, axis + 1:rnk]
    newdata = np.reshape(np.transpose(a, tuple(newdims)), (a.shape[axis], np.prod(a.shape, axis = 0) // a.shape[axis]))
    newdata = newdata.copy()
    
    # compute linear fit as least squared residuals
    x = np.arange(0, a.shape[axis], 1)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, newdata)[0]
    
    # remove the trend from the data along 1. axis
    for i in range(a.shape[axis]):
        newdata[i, ...] = newdata[i, ...] - (m*x[i] + c)
    
    # reshape back to original shape
    tdshape = np.take(a.shape, newdims, 0)
    ret = np.reshape(newdata, tuple(tdshape))
    vals = list(range(1,rnk))
    olddims = vals[:axis] + [0] + vals[axis:]
    ret = np.transpose(ret, tuple(olddims))
    
    # return detrended data and linear coefficient
    
    return ret, m
        


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
            
            data = v[:] # masked array - only land data, not ocean/sea
            self.data = data.data.copy() # get only data, not mask
            self.data[data.mask] = np.nan # filled masked values with NaNs
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

            self.data = v[:] - 273.15 # ERA is in Kelvins
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
            self.time = d.variables['time'][:] # hours since 1800-01-01 00:00
            self.time = self.time / 24.0 + date.toordinal(date(1800, 1, 1))
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            
            d.close()
            
            
            
    def load_station_data(self, filename, dataset = 'ECA-station', print_prog = True):
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
                        if row[1][-5] == ' ': # for stations with 2-digit SOUID
                            station = row[1][1:-13].lower()
                        elif row[1][-6] == ' ': # for stations with 3-digit SOUID
                            station = row[1][1:-14].lower()
                        elif row[1][-7] == ' ': # for stations with 4-digit SOUID
                            station = row[1][1:-15].lower()
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
            if print_prog:
                print("Station data from %s saved to structure. Shape of the data is %s" % (self.location, str(self.data.shape)))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            if self.missing.shape[0] != 0:
                print("** WARNING: There were some missing values! To be precise, %d missing values were found!" % (self.missing.shape[0]))
                  
                  
                  
    def copy_data(self):
        """
        Returns the copy of data.
        """              
        
        return self.data.copy()
                                            
                    
                    
    def select_date(self, date_from, date_to, apply_to_data = True):
        """
        Selects the date range - date_from is inclusive, date_to is exclusive. Input is date(year, month, day).
        """
        
        d_start = date_from.toordinal()
        d_to = date_to.toordinal()
        
        ndx = np.logical_and(self.time >= d_start, self.time < d_to)
        if apply_to_data:
            self.time = self.time[ndx] # slice time stamp
            self.data = self.data[ndx, ...] # slice data
        if self.missing is not None:
            missing_ndx = np.logical_and(self.missing >= d_start, self.missing < d_to)
            self.missing = self.missing[missing_ndx] # slice missing if exists
            
        return ndx
        
    
    
    def get_date_from_ndx(self, ndx):
        """
        Returns the date of the variable from given index.
        """
        
        return date.fromordinal(np.int(self.time[ndx]))
        
        
        
    def get_spatial_dims(self):
        """
        Returns the spatial dimensions of the data as list.
        """
        
        return list(self.data.shape[1:])
        
    
        
    def find_date_ndx(self, date):
        """
        Returns index which corresponds to the date. Returns None if the date is not contained in the data.
        """
        
        d = date.toordinal()
        pos = np.nonzero(self.time == d)
        if not np.all(np.isnan(pos)):
            return int(pos[0])
        else:
            return None
            
            
            
    def select_months(self, months, apply_to_data = True):
        """
        Subselects only certain months. Input as a list of months number.
        """
        
        ndx = filter(lambda i: date.fromordinal(int(self.time[i])).month in months, range(len(self.time)))
        
        if apply_to_data:
            self.time = self.time[ndx]
            self.data = self.data[ndx]
        
        return ndx
        
        
        
    def select_lat_lon(self, lats, lons):
        """
        Selects region in lat/lon. Input is for both [from, to], both are inclusive. If None, the dimension is not modified.
        """
        
        if self.lats is not None and self.lons is not None:
            if lats is not None:
                lat_ndx = np.nonzero(np.logical_and(self.lats >= lats[0], self.lats <= lats[1]))[0]
            else:
                lat_ndx = np.arange(len(self.lats))
                
            if lons is not None:
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
        
        if (self.missing is not None) and (self.missing.shape[0] != 0):
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
            

            
    def flatten_field(self, f = None):
        """
        Reshape the field to 2dimensions such that axis 0 is temporal and axis 1 is spatial.
        If f is None, reshape the self.data field, else reshape the f field.
        """        

        if f is None:
            if self.data.ndim == 3:
                self.data = np.reshape(self.data, (self.data.shape[0], np.prod(self.data.shape[1:])))
            else:
                raise Exception('Data field is already flattened!')

        elif f is not None:
            if f.ndim == 3:
                f = np.reshape(f, (f.shape[0], np.prod(f.shape[1:])))

                return f
            else:
                raise Exception('The field f is already flattened!')



    def reshape_flat_field(self, f = None):
        """
        Reshape flattened field to original time x lat x lon shape.
        If f is None, reshape the self.data field, else reshape the f field.
        """

        if f is None:
            if self.data.ndim == 2:
                new_shape = [self.data.shape[0]] + list((self.lats.shape[0], self.lons.shape[0]))
                self.data = np.reshape(self.data, new_shape)
            else:
                raise Exception('Data field is not flattened!')

        elif f is not None:
            if f.ndim == 2:
                new_shape = [f.shape[0]] + list((self.lats.shape[0], self.lons.shape[0]))
                f = np.reshape(f, new_shape)

                return f
            else:
                raise Exception('The field f is not flattened!')
                
                
                
    def get_data_of_precise_length(self, length = '16k', start_date = None, end_date = None, COPY = False):
        """
        Selects the data such that the length of the time series is exactly length.
        If COPY is True, it will replace the data and time, if False it will return them.
        If end_date is defined, it is exclusive.
        """
        
        if isinstance(length, int):
            ln = length
        elif 'k' in length:
            order = int(length[:-1])
            pow2list = np.array([np.power(2,n) for n in range(10,22)])
            ln = pow2list[np.where(order == pow2list/1000)[0][0]]
        else:
            raise Exception('Could not understand the length! Please type length as integer or as string like "16k".')
        
        if start_date is not None and self.find_date_ndx(start_date) is None:
            start_date = self.get_date_from_ndx(0)
        if end_date is not None and self.find_date_ndx(end_date) is None:
            end_date = self.get_date_from_ndx(-1)
        
        if end_date is None and start_date is not None:
            # from start date until length
            idx = self.find_date_ndx(start_date)
            data_temp = self.data[idx : idx + ln, ...].copy()
            time_temp = self.time[idx : idx + ln, ...].copy()
            idx_tuple = (idx, idx+ln)
            
        elif start_date is None and end_date is not None:
            idx = self.find_date_ndx(end_date)
            data_temp = self.data[idx - ln : idx, ...].copy()
            time_temp = self.time[idx - ln : idx, ...].copy()
            idx_tuple = (idx - ln, idx)
            
        else:
            raise Exception('You messed start / end date selection! Pick only one!')
            
        if COPY:
            self.data = data_temp.copy()
            self.time = time_temp.copy()
            return idx_tuple
            
        else:
            return data_temp, time_temp, idx_tuple



    def _shift_index_by_month(self, current_idx):
        """
        Returns the index in data shifted by month.
        """
        
        dt = date.fromordinal(self.time[current_idx])
        if dt.month < 12:
            mi = dt.month + 1
            y = dt.year
        else:
            mi = 1
            y = dt.year + 1
            
        return self.find_date_ndx(date(y, mi, dt.day))
        
            
            
    def get_monthly_data(self):
        """
        Converts the daily data to monthly means.
        """
        
        delta = self.time[1] - self.time[0]
        if delta == 1:
            # daily data
            day, mon, year = self.extract_day_month_year()
            monthly_data = []
            monthly_time = []
            # if first day of the data is not the first day of month - shift month
            # by one to start with the full month
            if day[0] != 1:
                mi = mon[0]+1 if mon[0] < 12 else 1
                y = year[0] if mon[0] < 12 else year[0] + 1
            else:
                mi = mon[0]
                y = year[0]
            start_idx = self.find_date_ndx(date(y, mi, 1))
            end_idx = self._shift_index_by_month(start_idx)
            while end_idx <= self.data.shape[0] and end_idx is not None:
                monthly_data.append(np.mean(self.data[start_idx : end_idx, ...], axis = 0))
                monthly_time.append(self.time[start_idx])
                start_idx = end_idx
                end_idx = self._shift_index_by_month(start_idx)
                if end_idx is None: # last piece, then exit the loop
                    monthly_data.append(np.mean(self.data[start_idx : , ...], axis = 0))
                    monthly_time.append(self.time[start_idx])
            self.data = np.array(monthly_data)
            self.time = np.array(monthly_time)                
        elif abs(delta - 30) < 3.0:
            # monhtly data
            raise Exception('The data are already monthly values.')
        else:
            raise Exception('Unknown temporal sampling in the field.')
            
            
        
    def average_to_daily(self):
        """
        Averages the 6-hourly values (e.g. ERA-40 basic sampling) into daily.
        """        
        
        delta = self.time[1] - self.time[0]
        if delta < 1:
            d = np.zeros([self.data.shape[0] / 4] + self.get_spatial_dims())
            t = np.zeros(self.time.shape[0] / 4)
            for i in range(d.shape[0]):
                d[i, ...] = np.mean(self.data[4*i : 4*i+3, ...], axis = 0)
                t[i] = self.time[4*i]
                
            self.data = d
            self.time = t.astype(np.int)
        
        else:
            raise Exception('No sub-daily values, you can average to daily only values with finer time sampling.')
        
        
        
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
                self.data, _ = nandetrend(self.data, axis = 0)
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
                self.data, _ = nandetrend(self.data, axis = 0)
                trend = data_copy - self.data
            else:
                trend = None
        else:
            raise Exception('Unknown temporal sampling in the field.')
            
        return seasonal_mean, seasonal_var, trend
        
        
        
    def return_seasonality(self, mean, var, trend):
        """
        Return the seasonality to the data.
        """
        
        if trend is not None:
            self.data += trend
        self.data *= var
        self.data += mean



    def center_data(self):
        """
        Reduces data time series to zero mean and unit variance (without respect for the seasons or temporal sampling). 
        """

        self.data -= nanmean(self.data, axis = 0)
        self.data /= nanstd(self.data, axis = 0, ddof = 1) 
        
        
        
        
def load_station_data(filename, start_date, end_date, anom, to_monthly = False, dataset = 'ECA-station'):
    """
    Data loader for station data.
    """
    
    print("[%s] Loading station data..." % (str(datetime.now())))
    path, name = split(filename)
    if path != '':
        path += "/"
        g = DataField(data_folder = path)
    else:
        g = DataField()
    g.load_station_data(name, dataset, print_prog = False)
    print("** loaded")
    g.select_date(start_date, end_date)
    if anom:
        print("** anomalising")
        g.anomalise()
    if to_monthly:
        g.get_monthly_data()
    day, month, year = g.extract_day_month_year()
    print("[%s] Data from %s loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), g.location, str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))
           
    return g



def load_NCEP_data_monthly(filename, varname, start_date, end_date, lats, lons, level, anom):
    """
    Data loader for monthly reanalyses data. 
    """

    print("[%s] Loading monthly NCEP/NCAR data..." % str(datetime.now()))
    path, name = split(filename)
    if path != '':
        path += "/"
        g = DataField(data_folder = path)
    else:
        g = DataField()
    g.load(name, varname, dataset = 'NCEP', print_prog = False)
    print("** loaded")
    g.select_date(start_date, end_date)
    g.select_lat_lon(lats, lons)
    if level is not None:
        g.select_level(level)
    if anom:
        print("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()

    print("[%s] NCEP data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))

    return g
    
    
    
def load_ERA_data_daily(filename, varname, start_date, end_date, lats, lons, anom, parts = 1, logger_function = None):
    """
    Data loader for daily ERA-40 / ERA-Interim data.
    If more than one file, filename should be all letters they have got in common without suffix.
    """
    
    if logger_function is None:
        def logger(msg):
            print("[%s] %s" % (str(datetime.now()), msg))
    else:
        logger = logger_function
        
    logger("Loading daily ERA-40 / ERA-Interim data...")
    
    # if in one file, just load it
    if parts == 1:
        path, name = split(filename)
        if path != '':
            path += '/'
            g = DataField(data_folder = path)
        else:
            g = DataField()
        g.load(name, varname, dataset = 'ERA-40', print_prog = False)
    
    # if in more files, find them all and load them
    else:
        fnames = []
        glist = []
        Ndays = 0
        path, name = split(filename)
        if path != '':
            path += '/'
        else:
            path = '../data'
        for root, _, files in os.walk(path):
            if root == path:
                for f in files:
                    if name in f:
                        fnames.append(f)
        if parts != len(fnames): 
            logger("Something went wrong since %d files matching your filename were found instead of %d." % (len(fnames), parts))
            raise Exception('Check your files and enter correct number of files you want to load.')
        for f in fnames:
            g = DataField(data_folder = path + '/')                
            g.load(f, varname, dataset = 'ERA-40', print_prog = False)
            Ndays += g.time.shape[0]
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
        
    if not np.all(np.unique(g.time) == g.time):
        logger('**WARNING: Some fields are overlapping, trying to fix this... (please note: experimental feature)')
        doubles = []
        for i in range(g.time.shape[0]):
            if np.where(g.time == g.time[i])[0].shape[0] == 1:
                # if there is one occurence of time value do nothing
                pass
            else:
                # remember the indices of other occurences
                doubles.append(np.where(g.time == g.time[i])[0][1:])
        logger("... found %d multiple values (according to the time field)..." % (len(doubles)/4))
        delete_mask = np.squeeze(np.array(doubles)) # mask with multiple indices
        # time
        g.time = np.delete(g.time, delete_mask)
        # data
        g.data = np.delete(g.data, delete_mask, axis = 0)
        
        
    logger("** loaded")
    g.select_date(start_date, end_date)
    g.select_lat_lon(lats, lons)
    g.average_to_daily()
    if anom:
        logger("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()
    logger("ERA-40 / ERA-Interim data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1])) 
           
    return g


def load_ECA_D_data_daily(filename, varname, start_date, end_date, lats, lons, anom, logger_function = None):
    """
    Data loader for daily ECA&D reanalysis data.
    """

    if logger_function is None:
        def logger(msg):
            print("[%s] %s" % (str(datetime.now()), msg))
    else:
        logger = logger_function

    logger("Loading daily ECA&D data...")
    path, name = split(filename)
    if path != '':
        path += "/"
        g = DataField(data_folder = path)
    else:
        g = DataField()
    g.load(name, varname, dataset = 'ECA-reanalysis', print_prog = False)
    logger("** loaded")
    g.select_date(start_date, end_date)
    g.select_lat_lon(lats, lons)
    if anom:
        logger("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()

    logger("ECA&D data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(g.data.shape), day[0], month[0], 
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
    if level is not None:
        g.select_level(level)
    if anom:
        print("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()

    print("[%s] NCEP data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))
           
    return g
    
    
    
def load_sunspot_data(filename, start_date, end_date, smoothed = False):
    """
    Data loader for ASCII file of sunspot number from Royal Observatory of Belgium.
    """
    
    path, name = split(filename)
    if path != '':
        path += "/"
        g = DataField(data_folder = path)
    else:
        g = DataField()
    with open(g.data_folder + filename, 'rb') as f:
        time = []
        data = []
        reader = csv.reader(f)
        for row in reader:
            year = int(row[0][:4])
            month = int(row[0][4:6])
            day = 1
            time.append(date(year, month, day).toordinal())
            if not smoothed:
                data.append(float(row[0][19:24]))
            else:
                if row[0][27:32] == '':
                    data.append(np.nan)
                else:
                    data.append(float(row[0][27:32]))
    
    g.data = np.array(data)
    g.time = np.array(time)
    g.location = 'The Sun'
    print("** loaded")
    g.select_date(start_date, end_date)
    _, month, year = g.extract_day_month_year()
    
    print("[%s] %s data loaded with shape %s. Date range is %d/%d - %d/%d inclusive." 
        % (str(datetime.now()), 'Sunspot' if not smoothed else 'Smoothed sunspot', str(g.data.shape), month[0], 
           year[0], month[-1], year[-1]))
           
    return g



def load_AAgeomag_data(filename, start_date, end_date, anom, daily = False):
    """
    Data loader for ASCII file of AA index -- geomagnetic field.
    """

    from dateutil.relativedelta import relativedelta
    
    path, name = split(filename)
    if path != '':
        path += "/"
        g = DataField(data_folder = path)
    else:
        g = DataField()
    raw_data = np.loadtxt(g.data_folder + filename) # first column is continous year and second is actual data
    g.data = np.array(raw_data[:, 1])
    time = []
    
    # use time iterator to go through the dates
    y = int(np.modf(raw_data[0, 0])[1]) 
    if np.modf(raw_data[0, 0])[0] == 0:
        starting_date = date(y, 1, 1)
    if daily:
        delta = timedelta(days = 1)
    else:
        delta = relativedelta(months = +1)
    d = starting_date
    while len(time) < raw_data.shape[0]:
        time.append(d.toordinal())
        d += delta
    g.time = np.array(time)
    g.location = 'The Earth'

    print("** loaded")
    g.select_date(start_date, end_date)
    if anom:
        print("** anomalising")
        g.anomalise()
    _, month, year = g.extract_day_month_year()
    
    print("[%s] AA index data loaded with shape %s. Date range is %d/%d - %d/%d inclusive." 
        % (str(datetime.now()), str(g.data.shape), month[0], 
           year[0], month[-1], year[-1]))
           
    return g
    
    
    
def load_bin_data(filename, start_date, end_date, anom):
    """
    Data loader for daily binned data.
    """

    import cPickle
    
    path, name = split(filename)
    if path == '':
        filename = "../data/" + filename
    with open(filename, 'rb') as f:
        g = cPickle.load(f)['g']
        
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
