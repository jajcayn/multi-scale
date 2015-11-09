"""
created on Jan 29, 2014

@author: Nikola Jajcay, jajcay(at)cs.cas.cz
based on class by Martin Vejmelka -- https://github.com/vejmelkam/ndw-climate --

last update on Aug 19, 2015
"""

import numpy as np
from datetime import date, timedelta, datetime
import csv
from os.path import split
import os


        
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
    
    def __init__(self, data_folder = '', data = None, lons = None, lats = None, time = None):
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
        self.nans = False
        self.cos_weights = None
        
        
        
    def load(self, filename = None, variable_name = None, dataset = 'ECA-reanalysis', print_prog = True):
        """
        Loads geophysical data from netCDF file for reanalysis or from text file for station data.
        Now supports following datasets: (dataset - keyword passed to function)
            ECA&D E-OBS gridded dataset reanalysis - 'ECA-reanalysis'
            ECMWF ERA-40 gridded reanalysis - 'ERA-40'
            NCEP/NCAR Reanalysis 1 - 'NCEP'
        """

        from netCDF4 import Dataset
        
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
            if np.any(np.isnan(self.data)):
                self.nans = True
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
                print("The first data value is from %s and the last is from %s" % (str(self.get_date_from_ndx(0)), str(self.get_date_from_ndx(-1))))
                print("Default temporal sampling in the data is %.2f day(s)" % (np.nanmean(np.diff(self.time))))
                if np.any(np.isnan(self.data)):
                    print("The data contains NaNs! All methods are compatible with NaNs, just to let you know!")
            
            d.close()     
                    
        if dataset == 'ERA-40':
            d = Dataset(self.data_folder + filename, 'r')
            v = d.variables[variable_name]

            self.data = v[:]
            self.lons = d.variables['longitude'][:]
            self.lats = d.variables['latitude'][:]
            self.time = d.variables['time'][:] # hours since 1900-01-01 00:00
            self.time = self.time / 24.0 + date.toordinal(date(1900, 1, 1))
            if np.any(np.isnan(self.data)):
                self.nans = True
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
                print("The first data value is from %s and the last is from %s" % (str(self.get_date_from_ndx(0)), str(self.get_date_from_ndx(-1))))
                print("Default temporal sampling in the data is %.2f day(s)" % (np.nanmean(np.diff(self.time))))
                if np.any(np.isnan(self.data)):
                    print("The data contains NaNs! All methods are compatible with NaNs, just to let you know!")
            
            d.close()
            
        if dataset == 'NCEP':
            d = Dataset(self.data_folder + filename, 'r')
            v = d.variables[variable_name]
            
            data = v[:] # masked array - only land data, not ocean/sea
            if isinstance(data, np.ma.masked_array):             
                self.data = data.data.copy() # get only data, not mask
                self.data[data.mask] = np.nan # filled masked values with NaNs
            else:
                self.data = data
            self.lons = d.variables['lon'][:]
            self.lats = d.variables['lat'][:]
            if 'level' in d.variables.keys():
                self.level = d.variables['level'][:]
            self.time = d.variables['time'][:] # hours or days since some date
            date_since = self._parse_time_units(d.variables['time'].units)
            if "hours" in d.variables['time'].units:
                self.time = self.time / 24.0 + date.toordinal(date_since)
            elif "days" in d.variables['time'].units:
                self.time += date.toordinal(date_since)
            if np.any(np.isnan(self.data)):
                self.nans = True
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
                print("The first data value is from %s and the last is from %s" % (str(self.get_date_from_ndx(0)), str(self.get_date_from_ndx(-1))))
                print("Default temporal sampling in the data is %.2f day(s)" % (np.nanmean(np.diff(self.time))))
                if np.any(np.isnan(self.data)):
                    print("The data contains NaNs! All methods are compatible with NaNs, just to let you know!")
            
            d.close()



    def _parse_time_units(self, time_string):
        """
        Parses time units from netCDF file, returns date since the record.
        """

        date_split = time_string.split('-')
        y = date_split[0][-4:]
        m = ("%02d" % int(date_split[1]))
        d = ("%02d" % int(date_split[2][:2]))

        return datetime.strptime("%s-%s-%s" % (y, m, d), '%Y-%m-%d')
            
            
            
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



    def copy(self):
        """
        Returns a copy of DataField with data, lats, lons and time fields.
        """

        copied = DataField()
        copied.data = self.data.copy()
        copied.lats = self.lats.copy()
        copied.lons = self.lons.copy()
        copied.time = self.time.copy()
        copied.nans = self.nans

        return copied   
                                            
                    
                    
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
        
        return list(self.data.shape[-2:])
        
    
        
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
            self.data = self.data[ndx, ...]
        
        return ndx
        
        
        
    def select_lat_lon(self, lats, lons, apply_to_data = True):
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
            
            if apply_to_data:
                d = self.data
                d = d[..., lat_ndx, :]
                self.data = d[..., lon_ndx]
                self.lats = self.lats[lat_ndx]
                self.lons = self.lons[lon_ndx]

                if np.any(np.isnan(self.data)):
                    self.nans = True
                else:
                    self.nans = False
            
            return lat_ndx, lon_ndx

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



    def latitude_cos_weights(self):
        """
        Returns a grid with scaling weights based on cosine of latitude.
        """
        
        if self.cos_weights != None:
            return self.cos_weights

        cos_weights = np.zeros(self.get_spatial_dims())
        for ndx in range(self.lats.shape[0]):
            cos_weights[ndx, :] = np.cos(self.lats[ndx] * np.pi/180.) ** 0.5

        self.cos_weights = cos_weights
        return cos_weights

        
        
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
        Should only be used with single-level data.
        """        

        if f is None:
            if self.data.ndim == 3:
                self.data = np.reshape(self.data, (self.data.shape[0], np.prod(self.data.shape[1:])))
            else:
                raise Exception('Data field is already flattened, multi-level or only temporal (e.g. station)!')

        elif f is not None:
            if f.ndim == 3:
                f = np.reshape(f, (f.shape[0], np.prod(f.shape[1:])))

                return f
            else:
                raise Exception('The field f is already flattened, multi-level or only temporal (e.g. station)!')



    def reshape_flat_field(self, f = None):
        """
        Reshape flattened field to original time x lat x lon shape.
        If f is None, reshape the self.data field, else reshape the f field.
        Supposes single-level data.
        """

        if f is None:
            if self.data.ndim == 2:
                new_shape = [self.data.shape[0]] + list((self.lats.shape[0], self.lons.shape[0]))
                self.data = np.reshape(self.data, new_shape)
            else:
                raise Exception('Data field is not flattened, is multi-level or is only temporal (e.g. station)!')

        elif f is not None:
            if f.ndim == 2:
                new_shape = [f.shape[0]] + list((self.lats.shape[0], self.lons.shape[0]))
                f = np.reshape(f, new_shape)

                return f
            else:
                raise Exception('The field f is not flattened, is multi-level or is only temporal (e.g. station)!')
                
                
                
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
            data_temp = self.data[idx - ln + 1 : idx + 1, ...].copy()
            time_temp = self.time[idx - ln + 1 : idx + 1, ...].copy()
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
                monthly_data.append(np.nanmean(self.data[start_idx : end_idx, ...], axis = 0))
                monthly_time.append(self.time[start_idx])
                start_idx = end_idx
                end_idx = self._shift_index_by_month(start_idx)
                if end_idx is None: # last piece, then exit the loop
                    monthly_data.append(np.nanmean(self.data[start_idx : , ...], axis = 0))
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
        Averages the sub-daily values (e.g. ERA-40 basic sampling is 6 hours) into daily.
        """        
        
        delta = self.time[1] - self.time[0]
        if delta < 1:
            n_times = int(1 / delta)
            d = np.zeros_like(self.data)
            d = np.delete(d, slice(0, (n_times-1) * d.shape[0]/n_times), axis = 0)
            t = np.zeros(self.time.shape[0] / n_times)
            for i in range(d.shape[0]):
                d[i, ...] = np.nanmean(self.data[n_times*i : n_times*i+(n_times-1), ...], axis = 0)
                t[i] = self.time[n_times*i]
                
            self.data = d
            self.time = t.astype(np.int)
        
        else:
            raise Exception('No sub-daily values, you can average to daily only values with finer time sampling.')



    def _ascending_descending_lat_lons(self, lats = True, lons = False, direction = 'asc'):
        """
        Transforms the data (and lats and lons) so that they have strictly ascending (direction = 'asc')
        or descending (direction = 'des') order. (Needed for interpolation).
        Returns True if manipulation took place.
        """

        lat_flg, lon_flg = False, False
        if np.all(np.diff(self.lats) < 0) and lats and direction == 'asc':
            self.lats = self.lats[::-1]
            self.data = self.data[..., ::-1, :]
            lat_flg = True
        elif np.all(np.diff(self.lats) > 0) and lats and direction == 'des':
            self.lats = self.lats[::-1]
            self.data = self.data[..., ::-1, :]
            lat_flg = True

        if np.all(np.diff(self.lons) < 0) and lons and direction == 'asc':
            self.lons = self.lons[::-1]
            self.data = self.data[..., ::-1]
            lon_flg = True
        elif np.all(np.diff(self.lons) > 0) and lons and direction == 'des':
            self.lons = self.lons[::-1]
            self.data = self.data[..., ::-1]
            lon_flg = True

        return lat_flg, lon_flg



    def subsample_spatial(self, lat_to, lon_to, start, average = False):
        """
        Subsamples the data in the spatial sense to grid "lat_to" x "lon_to" in degress.
        Start is starting point for subsampling in degrees as [lat, lon]
        If average is True, the subsampling is due to averaging the data -- using SciPy's spline
        interpolation on the rectangle. The interpolation is done for each time step and level 
        independently.
        If average is False, the subsampling is just subsampling certain values.
        """

        if self.lats is not None and self.lons is not None:
            delta_lats = np.abs(self.lats[1] - self.lats[0])
            delta_lons = np.abs(self.lons[1] - self.lons[0])
            if lat_to % delta_lats == 0 and lon_to % delta_lons == 0:
                lat_ndx = int(lat_to // delta_lats)
                lon_ndx = int(lon_to // delta_lons)
                start_lat_ndx = np.where(self.lats == start[0])[0]
                start_lon_ndx = np.where(self.lons == start[1])[0]
                if start_lon_ndx.size == 1 and start_lat_ndx.size == 1:
                    start_lat_ndx = start_lat_ndx[0]
                    start_lon_ndx = start_lon_ndx[0]
                    if not average:
                        self.lats = self.lats[start_lat_ndx::lat_ndx]
                        self.lons = self.lons[start_lon_ndx::lon_ndx]
                        d = self.data
                        d = d[..., start_lat_ndx::lat_ndx, :]
                        self.data = d[..., start_lon_ndx::lon_ndx]

                    else:

                        from scipy.interpolate import RectBivariateSpline

                        lat_flg, lon_flg = self._ascending_descending_lat_lons(lats = True, lons = True, direction = 'asc')
                        # if data is single-level - create additional dummy dimension
                        if self.data.ndim == 3:
                            self.data = self.data[:, np.newaxis, :, :]
                        # fields for new lats / lons
                        new_lats = np.arange(start[0], self.lats[-1]+lat_to, lat_to)
                        new_lons = np.arange(start[1], self.lons[-1]+lon_to, lon_to)
                        d = np.zeros((list(self.data.shape[:2]) + [new_lats.shape[0], new_lons.shape[0]]))
                        for t in range(self.time.shape[0]):
                            for lvl in range(self.data.shape[1]):
                                int_scheme = RectBivariateSpline(self.lats, self.lons, self.data[t, lvl, ...])

                                d[t, lvl, ...] = int_scheme(new_lats, new_lons)

                        self.lats = new_lats
                        self.lons = new_lons
                        self.data = np.squeeze(d)

                        self._ascending_descending_lat_lons(lats = lat_flg, lons = lon_flg, direction = 'des')

                    if np.any(np.isnan(self.data)):
                        self.nans = True
                    else:
                        self.nans = False

                else:
                    raise Exception("Start lat and / or lon for subsampling does not exist in the data!")

            else:
                raise Exception("Subsampling lats only to multiples of %.2f and lons of %.2f" % (delta_lats, delta_lons))

        else:
            raise Exception("Cannot subsample station data, or data from one grid point!")



    def smoothing_running_avg(self, points, use_to_data = False):
        """
        Smoothing of time series using running average over points.
        If use_to_data is False, returns the data, otherwise rewrites the data in class.
        """

        d = np.zeros(([self.data.shape[0] - points + 1] + list(self.data.shape[1:])))
        
        for i in range(d.shape[0]):
            d[i, ...] = np.nanmean(self.data[i : i+points, ...], axis = 0)

        if use_to_data:
            self.data = d.copy()
            if points % 2 == 1:
            # time slicing when points is odd -- cut points//2 from the beginning and from the end
                self.time = self.time[points//2 : -points//2 + 1]
            else:
            # time slicing when points is even -- not sure where to cut
                pass
        else:
            return d



    def spatial_filter(self, filter_weights = [1, 2, 1], use_to_data = False):
        """
        Filters the data in spatial sense with weights filter_weights.
        If use_to_data is False, returns the data, otherwise rewrites the data in class.
        """

        if self.data.ndim == 3: 
            self.data = self.data[:, np.newaxis, :, :]

        mask = np.zeros(self.data.shape[-2:])
        filt = np.outer(filter_weights, filter_weights)

        mask[:filt.shape[0], :filt.shape[1]] = filt

        d = np.zeros((list(self.data.shape[:-2]) + [self.lats.shape[0] - len(filter_weights) + 1, self.lons.shape[0] - len(filter_weights) + 1]))

        for i in range(d.shape[-2]):
            for j in range(d.shape[-1]):
                avg_mask = np.array([[mask for kk in range(d.shape[1])] for ll in range(d.shape[0])])
                d[:, :, i, j] = np.average(self.data, axis = (2, 3), weights = avg_mask)
                mask = np.roll(mask, 1, axis = 1)
            # return mask to correct y position
            mask = np.roll(mask, len(filter_weights)-1, axis = 1)
            mask = np.roll(mask, 1, axis = 0)

        if use_to_data:
            self.data = np.squeeze(d).copy()
            # space slicing when length of filter is odd -- cut length//2 from the beginning and from the end
            if len(filter_weights) % 2 == 1:
                self.lats = self.lats[len(filter_weights)//2 : -len(filter_weights)//2 + 1]
                self.lons = self.lons[len(filter_weights)//2 : -len(filter_weights)//2 + 1]
            else:
            # space slicing when length of filter is even -- not sure where to cut
                pass
        else:
            return np.squeeze(d)



    def check_NaNs_only_spatial(self):
        """
        Returns True if the NaNs contained in the data are of spatial nature, e.g.
        masked land from sea dataset and so on.
        returns False if also there are some NaNs in the temporal sense.
        E.g. with spatial NaNs, the PCA could be still done, when filtering out the NaNs.
        """

        if self.nans:
            cnt = 0
            nangrid0 = np.isnan(self.data[0, ...])
            for t in range(1, self.time.shape[0]):
                if np.all(nangrid0) == np.all(np.isnan(self.data[t, ...])):
                    cnt += 1

            if self.time.shape[0] - cnt == 1:
                return True
            else:
                return False

        else:
            print("No NaNs in the data, nothing happened!")



    def filter_out_NaNs(self, field = None):
        """
        Returns flattened version of 3D data field without NaNs (e.g. for computational purposes).
        The data is just returned, self.data is still full 3D version. Returned data has first axis
        temporal and second combined spatial.
        Mask is saved for internal purposes (e.g. PCA) but also returned.
        """

        if self.nans:
            if self.check_NaNs_only_spatial():
                d = self.data.copy() if field is None else field
                d = self.flatten_field(f = d)
                mask = np.isnan(d)
                spatial_mask = mask[0, :]
                d_out_shape = (d.shape[0], d.shape[1] - np.sum(spatial_mask))
                d_out = d[~mask].reshape(d_out_shape)
                self.spatial_mask = spatial_mask

                return d_out, spatial_mask

            else:
                raise Exception("NaNs are also temporal, no way to filter them out!")

        else:
            print("No NaNs in the data, nothing happened!")



    def return_NaNs_to_data(self, field, mask = None):
        """
        Returns NaNs to the data and reshapes it to the original shape.
        Field has first axis temporal and second combined spatial.
        """

        if self.nans:
            if mask is not None or self.spatial_mask is not None:
                mask = mask if mask is not None else self.spatial_mask
                d_out = np.zeros((field.shape[0], mask.shape[0]))
                ndx = np.where(mask == False)[0]
                d_out[:, ndx] = field
                d_out[:, mask] = np.nan

                return self.reshape_flat_field(f = d_out)

            else:
                raise Exception("No mask given!")
        
        else:
            print("No NaNs in the data, nothing happened!")



    def pca_components(self, n_comps):
        """
        Estimate the PCA (EOF) components of geo-data.
        Shoud be used on single-level data.
        """

        if self.data.ndim == 3:
            from scipy.linalg import svd

            # reshape field so the first axis is combined spatial and second is temporal
            # if nans, filter-out
            if self.nans:
                d = self.filter_out_NaNs()[0]
            else:
                d = self.data.copy()
                d = self.flatten_field(f = d)
            d = d.transpose()

            # remove mean of each time series
            pca_mean = np.mean(d, axis = 0)
            self.pca_mean = pca_mean
            d -= self.pca_mean  

            U, s, V = svd(d, False, True, True)
            U *= s
            exp_var = (s ** 2) / self.time.shape[0]
            exp_var /= np.sum(exp_var)
            eofs = U[:, :n_comps]
            pcs = V[:n_comps, :]
            var = exp_var[:n_comps]

            eofs = eofs.transpose()
            if self.nans:
                eofs = self.return_NaNs_to_data(field = eofs)
            else:
                eofs = self.reshape_flat_field(f = eofs)

            return eofs, pcs, var

        else:
            raise Exception("PCA analysis cannot be used on multi-level data or only temporal (e.g. station) data!")



    def invert_pca(self, eofs, pcs, pca_mean = None):
        """
        Inverts the PCA and returns the original data.
        Suitable for modelling, pcs could be different than obtained from PCA.
        """

        if self.nans:
            e = self.filter_out_NaNs(field = eofs)[0]
        else:
            e = eofs.copy()
            e = self.flatten_field(f = e)
        e = e.transpose()

        pca_mean = pca_mean if pca_mean is not None else self.pca_mean

        recons = np.dot(e, pcs)
        recons += pca_mean

        recons = recons.transpose()
        if self.nans:
            recons = self.return_NaNs_to_data(field = recons)
        else:
            recons = self.reshape_flat_field(f = recons)

        return recons


        
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
                    avg = np.nanmean(self.data[sel, ...], axis = 0)
                    self.data[sel, ...] -= avg
        elif abs(delta - 30) < 3.0:
            # monthly data
            _, mon, _ = self.extract_day_month_year()
            for mi in range(1,13):
                sel = (mon == mi)
                if np.sum(sel) == 0:
                    continue
                avg = np.nanmean(self.data[sel, ...], axis = 0)
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
                    seasonal_mean[sel, ...] = np.nanmean(self.data[sel, ...], axis = 0)
                    self.data[sel, ...] -= seasonal_mean[sel, ...]
                    seasonal_var[sel, ...] = np.nanstd(self.data[sel, ...], axis = 0, ddof = 1)
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
                seasonal_mean[sel, ...] = np.nanmean(self.data[sel, ...], axis = 0)
                self.data[sel, ...] -= seasonal_mean[sel, ...]
                seasonal_var[sel, ...] = np.nanstd(self.data[sel, ...], axis = 0, ddof = 1)
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
        Centers data time series to zero mean and unit variance (without respect for the seasons or temporal sampling). 
        """

        self.data -= np.nanmean(self.data, axis = 0)
        self.data /= np.nanstd(self.data, axis = 0, ddof = 1) 



    def save_field(self, fname):
        """
        Saves entire Data Field to cPickle format.
        """

        import cPickle

        with open(fname, "wb") as f:
            cPickle.dump(self.__dict__, f, protocol = cPickle.HIGHEST_PROTOCOL)



    def load_field(self, fname):
        """
        Loads entire Data Field from pickled file.
        """

        import cPickle

        with open(fname, "rb") as f:
            data = cPickle.load(f)

        self.__dict__ = data
        
        
        
        
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
    raw_data = np.loadtxt(g.data_folder + name) # first column is continous year and second is actual data
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
