"""
created on Jan 29, 2014

@author: Nikola Jajcay, based on script by Martin Vejmelka> -- https://github.com/vejmelkam/ndw-climate --
"""

import numpy as np
from netCDF4 import Dataset
from datetime import date
import csv
import os


DATA_FOLDER = '../data/'


class DataField:
    """
    Class holds the time series of a geophysical field. The fields for reanalysis data are
    3-dimensional - two spatial and one temporal dimension.
    """
    
    def __init__(self, data = None, lons = None, lats = None, time = None):
        """
        Initialize either an empty data set or with given values.
        """
        self.data = data
        self.lons = lons
        self.lats = lats
        self.time = time
        
        
    def load(self, filename, variable_name, dataset = 'ECA-reanalysis'):
        """
        Load geophysical data from netCDF file for reanalysis or from text file for station data.
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
            
        if dataset == 'ECA-station':
            stations = {}
            for _, _, files in os.walk(DATA_FOLDER): # walk through the entire folder, omit root and dirs
                for name in files: # iterate through all files in DATA_FOLDER
                    if name == 'stations.txt':
                        pass
                
            
            
    def select_date(self, date_from, date_to):
        """
        Selects the date range - date_from is inclusive, date_to is exclusive. Input is date(year, month, day).
        """
        d_start = date_from.toordinal()
        d_to = date_to.toordinal()
        
        ndx = np.logical_and(self.time >= d_start, self.time < d_to)
        self.time = self.time[ndx] # slice time stamp
        self.data = self.data[ndx, ...] # slice data
        
        
    def find_date_ndx(self, date):
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
        Subselects only certain months. Input as a list of month numbers.
        """
        ndx = filter(lambda i: date.fromordinal(int(self.time[i])).month in months, range(len(self.time)))
        
        self.time = self.time[ndx]
        self.data = self.data[ndx]
        
        
    def select_lat_lon(self, lats, lons):
        """
        Select region in lat/lon. Input is for both [from, to], both are inclusive. If None, the dimension is not modified.
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
        