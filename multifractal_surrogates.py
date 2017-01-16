"""
##-----------------------------------------------------------------------------##
Script for generating multifractal surrogates. MF surrogates are independent
shuffling of the scale-specific coefficients, preserving so-called multifractal
structure of the data. Multifractal processes exhibit hierarchical information
flow from large to small time scales.

Written according to Palus, M. (2008): Bootstraping multifractals: Surrogate 
    data from random cascades on wavelet dyadic trees. Phys. Rev. Letters, 101.
    
    
    
requirements:: numpy package -- http://www.numpy.org/
               PyWavelets (pywt) package -- http://www.pybytes.com/pywavelets/

author: Nikola Jajcay, Institute of Computer Science, CAS.
"""


import numpy as np
import pywt


def multifractal_surrogate(ts, randomise_from_scale = 2, amplitude_adjust_surrogates = False):
    """
    Returns the multifractal surrogate realisation from given time series.
    
    Parameters
    ----------
    ts : numpy array
        1-D numpy array containing time series from which surrogate realisation
        should be taken. Time series length must be 2^n. 
    randomise_from_scale : int, optional
        Scale from which to randomise coefficients. Default is to not randomise
        first two scales (the two slowest frequencies).
    amplitude_adjust_surrogates : boolean, optional
        If True, returns amplitude adjusted surrogates, which are in fact original
        data sorted according to the generated surrogate data.
        
    Returns
    -------
    surr : numpy array
    
    """
    
    if ts.ndim > 1:
        raise Exception("Input time series should be 1-D numpy array!")
        
    n = int(np.log2(ts.shape[0]))
    n_real = np.log2(ts.shape[0])
    if n != n_real:
        raise Exception("Time series length must be a power of 2 (2^n)!")
        
    # get coefficient from discrete wavelet transform, 
    # it is a list of length n with numpy arrays as objects
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
        
        # sort shuffled coefficients
        idx = np.argsort(coeffs_tilde[j])

        # sort original coefficients
        coeffs[j] = np.sort(coeffs[j])
        
        # finally, rearange original coefficient according to coefficient with tilde
        coeffs_tmp = np.zeros_like(coeffs[j])
        coeffs_tmp[idx] = coeffs[j]
        shuffled_coeffs.append(coeffs_tmp)

    surr = pywt.waverec(shuffled_coeffs, 'db1')

    # if return amplitude adjusted surrogates
    if amplitude_adjust_surrogates:

        # sort generated surrogates
        idx = np.argsort(surr)

        # amplitude adjusted surrogates are original data sorted according to the surrogates
        ts = np.sort(ts)
        surr = np.zeros_like(ts)
        surr[idx] = ts

        
    return surr
        
    
    
    
    

