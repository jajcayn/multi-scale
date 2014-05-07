"""
created on Mar 4, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import DataField
import pywt




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
        
        if self.surr_data != None:
            if trend != None:
                self.surr_data += trend
            self.surr_data *= var
            self.surr_data += mean
        else:
            raise Exception("Surrogate data has not been created yet.")
        
        
        
    def get_surr(self):
        """
        Returns the surrogate data
        """
        
        if self.surr_data != None:
            return self.surr_data.copy()
        else:
            raise Exception("Surrogate data has not been created yet.")
        


    def construct_fourier_surrogates(self):
        """
        Constructs Fourier Transform (FT) surrogates (independent realizations which preserve
        linear structure and covariance structure)
        """
        
        if self.data != None:
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
            
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")
        
        
        
    def construct_fourier_surrogates_spatial(self):
        """
        Constructs Fourier Transform (FT) surrogates (independent realizations which preserve
        linear structure but not covariance structure - shuffles also along spatial dimensions)
        (should be also used with station data which has only temporal dimension)
        """
        
        if self.data != None:
            xf = np.fft.rfft(self.data, axis = 0)
            
            # same as above except generate random angles along all dimensions of input data
            angle = np.random.uniform(0, 2 * np.pi, xf.shape)
            
            angle[0, ...] = 0
            
            cxf = xf * np.exp(1j * angle)
            
            self.surr_data = np.fft.irfft(cxf, axis = 0)
            
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")
    
        
        
        
    def construct_multifractal_surrogates(self, randomise_from_scale = 2):
        """
        Constructs multifractal surrogates (independent shuffling of the scale-specific coefficients,
        preserving so-called multifractal structure - hierarchical process exhibiting information flow
        from large to small scales)
        written according to: Palus, M. (2008): Bootstraping multifractals: Surrogate data from random 
        cascades on wavelet dyadic trees. Phys. Rev. Letters, 101.
        """
        
        if self.data != None:
            
            if self.data.ndim > 1:
                num_lats = self.lats.shape[0]
                num_lons = self.lons.shape[0]
            else:
                num_lats = 1
                num_lons = 1
                self.data = self.data[:, np.newaxis, np.newaxis]
            
            self.surr_data = np.zeros_like(self.data)
            
            for lat in range(num_lats):
                for lon in range(num_lons):
                    n = int(np.log2(self.data.shape[0])) # time series length should be 2^n
                    n_real = np.log2(self.data.shape[0])
                    
                    if n != n_real:
                        # if time series length is not 2^n
                        raise Exception("Time series length must be power of 2 (2^n).")
                        break
                    
                    # get coefficient from discrete wavelet transform, 
                    # it is a list of length n with numpy arrays as every object
                    coeffs = pywt.wavedec(self.data[:, lat, lon], 'db1', level = n-1)
                    
                    # prepare output lists and append coefficients which will not be shuffled
                    coeffs_tilde = []
                    for j in range(randomise_from_scale):
                        coeffs_tilde.append(coeffs[j])
                
                    shuffled_coeffs = []
                    for j in range(randomise_from_scale):
                        shuffled_coeffs.append(coeffs[j])
                    
                    # run for each desired scale
                    for j in range(randomise_from_scale, len(coeffs)):
                        
                        # get multiplicators for scale j
                        multiplicators = np.zeros_like(coeffs[j])
                        for k in range(coeffs[j-1].shape[0]):
                            multiplicators[2*k] = coeffs[j][2*k] / coeffs[j-1][k]
                            multiplicators[2*k+1] = coeffs[j][2*k+1] / coeffs[j-1][k]
                       
                        # shuffle multiplicators in scale j randomly
                        coef = np.zeros_like(multiplicators)
                        multiplicators = np.random.permutation(multiplicators)
                        
                        # get coefficients with tilde according to a cascade
                        for k in range(coeffs[j-1].shape[0]):
                            coef[2*k] = multiplicators[2*k] * coeffs_tilde[j-1][k]
                            coef[2*k+1] = multiplicators[2*k+1] * coeffs_tilde[j-1][k]
                        coeffs_tilde.append(coef)
                        
                        # sort original coefficients
                        coeffs[j] = np.sort(coeffs[j])
                        
                        # sort shuffled coefficients
                        idx = np.argsort(coeffs_tilde[j])
                        
                        # finally, rearange original coefficient according to coefficient with tilde
                        shuffled_coeffs.append(coeffs[j][idx])
                        
                    # return randomised time series as inverse discrete wavelet transform
                    self.surr_data[:, lat, lon] = pywt.waverec(shuffled_coeffs, 'db1')
            
            # squeeze single-dimensional entries (e.g. station data)
            self.surr_data = np.squeeze(self.surr_data)
            self.data = np.squeeze(self.data)
            
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")
        

