%%
%Created on 16 Mar 2015
%@author: Nikola Jajcay -- jajcay@cs.cas.cz

%%-----------------------------------------------------------------------------%%
% Script for generating multifractal surrogates. MF surrogates are independent
% shuffling of the scale-specific coefficients, preserving so-called multifractal
% structure of the data. Multifractal processes exhibit hierarchical information
% flow from large to small time scales.

% Written according to Palus, M. (2008): Bootstraping multifractals: Surrogate 
%     data from random cascades on wavelet dyadic trees. Phys. Rev. Letters, 101.  

%%
function [surr] = multifractal_surrogate(ts, randomise_from_scale, amplitude_adjust_surrogates)

%
% tu bude popis presnejsi
%

% fill in defaults - amplitude adjust = False and randomise_from_scale = 2
if nargin() < 3
    amplitude_adjust_surrogates = 0;
end

if nargin() < 2
    randomise_from_scale = 2;
end

% check input time series
ts = squeeze(ts);
if (ndims(ts) ~= 2) or (all(squeeze(ts) == ts)) 
    error('Input time series should be 1-D vector!');
end

n = int16(log2(size(ts, 2)));
n_real = log2(size(ts, 2));

if n ~= n_real
    error('Time series length must be a power of 2 (2^n)!');
end




end