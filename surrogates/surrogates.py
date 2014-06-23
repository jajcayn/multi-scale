"""
created on Mar 4, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import DataField
import pywt
from var_model import VARModel



def _prepare_surrogates(a):
    i, j, order_range, crit, ts = a
    if np.any(np.isnan(ts)) == False:
        v = VARModel()
        v.estimate(ts, order_range, True, crit, None)
        r = v.compute_residuals(ts)
    else:
        v = None
        r = np.nan
    return (i, j, v, r) 
    
    
    
def _compute_AR_surrogates(a):
    i, j, res, model, num_tm_s = a
    r = np.zeros((num_tm_s, 1), dtype = np.float64)       
    if np.all(np.isnan(res)) == False:
        ndx = np.argsort(np.random.uniform(size = (num_tm_s,)))
        r[ndx, 0] = res

        ar_surr = model.simulate_with_residuals(r)[:, 0]
    else:
        ar_surr = np.nan
        
    return (i, j, ar_surr)
    
    
    
def _compute_FT_surrogates(a):
    i, j, data, angle = a
            
    # transform the time series to Fourier domain
    xf = np.fft.rfft(data, axis = 0)
     
    # randomise the time series with random phases       
    cxf = xf * np.exp(1j * angle)
    
    # return randomised time series in time domain
    ft_surr = np.fft.irfft(cxf, axis = 0)
    
    return (i, j, ft_surr)
    
    


def _compute_MF_surrogates(a):
    i, j, ts, randomise_from_scale = a
    if np.all(np.isnan(ts)) == False:
        n = int(np.log2(ts.shape[0])) # time series length should be 2^n
        n_real = np.log2(ts.shape[0])
        
        if n != n_real:
            # if time series length is not 2^n
            raise Exception("Time series length must be power of 2 (2^n).")
        
        # get coefficient from discrete wavelet transform, 
        # it is a list of length n with numpy arrays as every object
        coeffs = pywt.wavedec(ts, 'db1', level = n-1)
        
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
                if coeffs[j-1][k] == 0:
                    print("**WARNING: some zero coefficients in DWT transform!")
                    coeffs[j-1][k] = 1
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
        mf_surr = pywt.waverec(shuffled_coeffs, 'db1')

    else:
        mf_surr = np.nan

    return (i, j, mf_surr)




class SurrogateField(DataField):
    """
    Class holds geofield of surrogate data and can construct surrogates.
    """
    
    def __init__(self):
        DataField.__init__(self)
        self.surr_data = None
        self.model_grid = None
        

        
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
        


    def construct_fourier_surrogates(self, pool = None):
        """
        Constructs Fourier Transform (FT) surrogates (independent realizations which preserve
        linear structure and covariance structure)
        """
        
        if self.data != None:
            
            if pool == None:
                map_func = map
            else:
                map_func = pool.map
                
            # generate uniformly distributed random angles
            a = np.fft.rfft(np.random.rand(self.data.shape[0]), axis = 0)
            angle = np.random.uniform(0, 2 * np.pi, (a.shape[0],))
            
            # set the slowest frequency to zero, i.e. not to be randomised
            angle[0] = 0
            del a
            
            job_data = [ (i, j, self.data[:, i, j], angle) for i in range(self.lats.shape[0]) for j in range(self.lons.shape[0]) ]
            job_results = map_func(_compute_FT_surrogates, job_data)
            
            self.surr_data = np.zeros_like(self.data)
            
            for i, j, surr in job_results:
                self.surr_data[:, i, j] = surr
           
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")
        
        
        
    def construct_fourier_surrogates_spatial(self, pool = None):
        """
        Constructs Fourier Transform (FT) surrogates (independent realizations which preserve
        linear structure but not covariance structure - shuffles also along spatial dimensions)
        (should be also used with station data which has only temporal dimension)
        """
        
        if self.data != None:
            
            if pool == None:
                map_func = map
            else:
                map_func = pool.map
            
            # same as above except generate random angles along all dimensions of input data
            a = np.fft.rfft(np.random.rand(self.data.shape[0]), axis = 0)
            angle = np.random.uniform(0, 2 * np.pi, (a.shape[0], self.lats.shape[0], self.lons.shape[0]))
            angle[0, ...] = 0
            del a
            
            job_data = [ (i, j, self.data[:, i, j], angle[:, i, j]) for i in range(self.lats.shape[0]) for j in range(self.lons.shape[0]) ]
            job_results = map_func(_compute_FT_surrogates, job_data)
            
            self.surr_data = np.zeros_like(self.data)
            
            for i, j, surr in job_results:
                self.surr_data[:, i, j] = surr
            
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")
    
        
        
    def construct_multifractal_surrogates(self, pool = None, randomise_from_scale = 2):
        """
        Constructs multifractal surrogates (independent shuffling of the scale-specific coefficients,
        preserving so-called multifractal structure - hierarchical process exhibiting information flow
        from large to small scales)
        written according to: Palus, M. (2008): Bootstraping multifractals: Surrogate data from random 
        cascades on wavelet dyadic trees. Phys. Rev. Letters, 101.
        """
        
        if self.data != None:

            if pool == None:
                map_func = map
            else:
                map_func = pool.map
            
            if self.data.ndim > 1:
                num_lats = self.lats.shape[0]
                num_lons = self.lons.shape[0]
            else:
                num_lats = 1
                num_lons = 1
                self.data = self.data[:, np.newaxis, np.newaxis]
            
            self.surr_data = np.zeros_like(self.data)

            job_data = [ (i, j, self.data[:, i, j], randomise_from_scale) for i in range(num_lats) for j in range(num_lons) ]
            job_results = map_func(_compute_MF_surrogates, job_data)
            
            for i, j, surr in job_results:
                self.surr_data[:, i, j] = surr
            
            # squeeze single-dimensional entries (e.g. station data)
            self.surr_data = np.squeeze(self.surr_data)
            self.data = np.squeeze(self.data)
            
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")
        


    def prepare_AR_surrogates(self, pool = None, order_range = [1, 1], crit = 'sbc'):
        """
        Prepare for generating AR(k) surrogates by identifying the AR model and computing
        the residuals. Adapted from script by Vejmelka -- https://github.com/vejmelkam/ndw-climate
        """
        
        if self.data != None:
            
            if pool == None:
                map_func = map
            else:
                map_func = pool.map
                
            if self.data.ndim > 1:
                num_lats = self.lats.shape[0]
                num_lons = self.lons.shape[0]
            else:
                num_lats = 1
                num_lons = 1
                self.data = self.data[:, np.newaxis, np.newaxis]
            num_tm = self.time.shape[0]
                
            job_data = [ (i, j, order_range, crit, self.data[:, i, j]) for i in range(num_lats) for j in range(num_lons) ]
            job_results = map_func(_prepare_surrogates, job_data)
            max_ord = 0
            for r in job_results:
                if r[2] is not None and r[2].order() > max_ord:
                    max_ord = r[2].order()
            num_tm_s = num_tm - max_ord
            
            self.model_grid = np.zeros((num_lats, num_lons), dtype = np.object)
            self.residuals = np.zeros((num_tm_s, num_lats, num_lons), dtype = np.float64)
    
            for i, j, v, r in job_results:
                self.model_grid[i, j] = v
                if v is not None:
                    self.residuals[:, i, j] = r[:num_tm_s, 0]
                else:
                    self.residuals[:, i, j] = np.nan
    
            self.max_ord = max_ord
            
            self.data = np.squeeze(self.data)
            
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")
        
        
        
    def construct_surrogates_with_residuals(self, pool = None):
        """
        Constructs a new surrogate time series from AR(k) model.
        Adapted from script by Vejmelka -- https://github.com/vejmelkam/ndw-climate
        """
        
        if self.model_grid != None:
            
            if pool == None:
                map_func = map
            else:
                map_func = pool.map
            
            if self.data.ndim > 1:
                num_lats = self.lats.shape[0]
                num_lons = self.lons.shape[0]
            else:
                num_lats = 1
                num_lons = 1
            num_tm_s = self.time.shape[0] - self.max_ord
            
            job_data = [ (i, j, self.residuals[:, i, j], self.model_grid[i, j], num_tm_s) for i in range(num_lats) for j in range(num_lons) ]
            job_results = map_func(_compute_AR_surrogates, job_data)
            
            self.surr_data = np.zeros((num_tm_s, num_lats, num_lons))
            
            for i, j, surr in job_results:
                self.surr_data[:, i, j] = surr
                    
            self.surr_data = np.squeeze(self.surr_data)

        else:
           raise Exception("The AR(k) model is not simulated yet. First prepare surrogates!") 
