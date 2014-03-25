"""
created on Mar 4, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import DataField




class SurrogateField(DataField):
    """
    Class holds geofield of surrogate data and can construct surrogates.
    """
    
    def __init__(self):
        DataField.__init__(self)
        self.surr_data = None
        

        
    def copy_field(self, field):
        """
        Makes a copy of another DataField
        """
        
        self.data = field.data.copy()
        if field.lons != None:
            self.lons = field.lons.copy()
        else:
            self.lons = None
        if field.lats != None:
            self.lats = field.lats.copy()
        else:
            self.lats = None
        self.time = field.time.copy()
        
        
        
    def add_seasonality(self, mean, var, trend):
        """
        Adds seasonality to surrogates if there were constructed from deseasonalised
        and optionally detrended data.
        """
        
        if trend != None:
            self.surr_data += trend
        self.surr_data *= var
        self.surr_data += mean
        
        
        
    def get_surr(self):
        """
        Returns the surrogate data
        """
        
        return self.surr_data.copy()
        


    def construct_fourier_surrogates(self):
        """
        Constructs Fourier Transform (FT) surrogates (independent realizations which preserve
        linear structure and covariance structure)
        """
        
        # transform the time series to Fourier domain
        xf = np.fft.rfft(self.data, axis = 0)
        
        # generate uniformly distributed random angles
        angle = np.random.uniform(0, 2 * np.pi, (xf.shape[0],))
        
        # set the slowest frequency to zero, i.e. not to be randomised
        angle[0] = 0
        
        # randomise the time series with random phases
        cxf = xf * np.exp(1j * angle[:, np.newaxis, np.newaxis])
        
        # return randomised time series in time domain
        self.surr_data = np.fft.irfft(cxf, axis = 0)
        
        
        
    def construct_fourier_surrogates_spatial(self):
        """
        Constructs Fourier Transform (FT) surrogates (independent realizations which preserve
        linear structure but not covariance structure - shuffles also along spatial dimensions)
        (should be also used with station data which has only temporal dimension)
        """
        
        xf = np.fft.rfft(self.data, axis = 0)
        
        # same as above except generate random angles along all dimensions of input data
        angle = np.random.uniform(0, 2 * np.pi, xf.shape)
        
        angle[0, ...] = 0
        
        cxf = xf * np.exp(1j * angle)
        
        self.surr_data = np.fft.irfft(cxf, axis = 0)

