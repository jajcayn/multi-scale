"""
created on Feb 3, 2014

@author: Nikola Jajcay, 
inspired by A Practical Guide to Wavelet Analysis by Ch. Torrence and G. Compo
-- http://paos.colorado.edu/research/wavelets/ --
"""

import numpy as np
from scipy.fftpack import fft, ifft


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
    
    
    
def continous_wavelet(X, dt, pad = False, **kwargs):
    """
    Computes the wavelet transform of the vector X, with sampling rate dt.
    
    inputs:
    X - the time series, numpy array
    dt - sampling time of dt
    pad - if True, pad time series with 0 to get len(X) up to the next higher power of 2. It speeds up the FFT.
    --- kwargs ---
    dj - the spacing between discrete scales.
    s0 - the smallest scale of the wavelet
    j1 - the number of scales minus one. Scales range from s0 up to s0 * 2^(j1+dj) to give a total of j1+1 scales. 
    k0 - Morlet wavelet parameter, wavenumber
    """
    # map arguments
    if 'dj' in kwargs:
        dj = kwargs['dj']
    else:
        dj = 0.25
    if 's0' in kwargs:
        s0 = kwargs['s0']
    else:
        s0 = 2 * dt
    if 'j1' in kwargs:
        j1 = kwargs['j1']
    else:
        j1 = np.fix(np.log(len(X)*dt/s0) / np.log(2)) / dj
    if 'k0' in kwargs:
        k0 = kwargs['k0']
    else:
        k0 = 6.
    
    n1 = len(X)
    
    Y = X - np.mean(X)
    
    # padding, if needed
    if pad:
        base2 = np.fix(np.log(n1)/np.log(2) + 0.4999999) # power of 2 nearest to len(X)
        Y = np.concatenate((Y, np.zeros(np.power(2, (base2+1) - n1))))
    n = len(Y)
    
    # wavenumber array
    k = np.arange(1, np.fix(n/2) + 1)
    k *= (2. * np.pi) / (n * dt)
    k_minus = -k[np.fix(n-1)/2 - 1::-1]
    k = np.concatenate((np.array([0.]), k, k_minus))
    
    # compute FFT of the (padded) time series
    f = fft(Y)
    
    # construct scale array and empty period & wave arrays
    scale = np.array( [s0 * np.power(2, x*dj) for x in range(0,j1+1)] )
    period = scale
    wave = np.zeros((j1+1, n), dtype = np.complex)
    
    # loop through scales and compute tranform
    for i in range(j1+1):
        daughter, fourier_factor, _ = morlet(k, scale[i], k0)
        wave[i, :] = ifft(f * daughter)
        
    period = fourier_factor * scale
    wave = wave[:, :n1]
    
    return wave, period, scale
    