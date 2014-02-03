"""
created on Feb 3, 2014

@author: Nikola Jajcay, inspired by A Practical Guide to Wavelet Analysis by Ch. Torrence and G. Compo
-- http://paos.colorado.edu/research/wavelets/ --
"""

import numpy as np


def morlet(k, scale, k0 = 6.):
    """
    Returns the Morlet wavelet function as a function of Fourier frequency,
    used for the wavelet transform in Fourier space.
    
    Morlet wavelet: psi(x) = pi^(-1/4) * exp(i*k0*x) * exp(-x^2 / 2)
    
    inputs:
    k - numpy array with Fourier frequencies at which to calculate the wavelet
    scale - the wavelet scale
    k0 - wavenumber
    """
    
    exponent = - np.power((scale * k - k0),2) / 2. * (k > 0.)
    norm = np.sqrt(scale * k[1]) * (np.power(np.pi, -0.25)) * np.sqrt(len(k))
    output = norm * np.exp(exponent)
    output *= (k > 0.)
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
    coi = fourier_factor / np.sqrt(2.)
    
    return output, fourier_factor, coi
    
    
    
def wavelet(X, dt, pad = False, dj = 0.25, s0 = 2*dt, j1 = np.fix(np.log(len(X)*dt/s0) / np.log(2)) / dj, k0):
    """
    Computes the wavelet transform of the vector X, with sampling rate dt.
    
    inputs:
    X - the time series, numpy array
    dt - sampling time of dt
    pad - if True, pad time series with 0 to get len(X) up to the next higher power of 2. It speeds up the FFT.
    dj - the spacing between discrete scales.
    s0 - the smallest scale of the wavelet
    j1 - the number of scales minus one. Scales range from s0 up to s0 * 2^(j1+dj) to give a total of j1+1 scales. 
    k0 - Morlet wavelet parameter, wavenumber
    """
    
    n1 = len(X)
    
    Y = X - np.mean(X)
    # padding, if needed
    if pad:
        base2 = np.fix(np.log(n1)/np.log(2) + 0.4999999) # power of 2 nearest to len(X)
        Y = np.concatenate((Y, np.zeros(np.power(2, (base2+1) - n1)))
