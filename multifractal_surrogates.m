%%
% created on 16 Mar 2015
% author: Nikola Jajcay -- jajcay@cs.cas.cz

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
%     Returns the multifractal surrogate realisation from given time series.
%     
%     Parameters
%     ----------
%     ts : vector
%         matlab (1, length) - column vector containing time series from which surrogate realisation
%         should be taken. Time series length must be 2^n. 
%     randomise_from_scale : int, optional
%         Scale from which to randomise coefficients. Default is to not randomise
%         first two scales (the two slowest frequencies).
%     amplitude_adjust_surrogates : boolean, optional
%         If True, returns amplitude adjusted surrogates, which are in fact original
%         data sorted according to the generated surrogate data.
%         
%     Returns
%     -------
%     surr : (1, length) - column vector
%

addpath('matlab_dwt');

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

% create filter bank coefficients
[af, sf] = db1;

% compute discrete wavelet
coeffs = dwt(ts, n-1, af);

% prepare output lists and append coefficients which will not be shuffled
coeffs_tilde = {};
for j=size(coeffs, 2):-1:size(coeffs, 2) - randomise_from_scale
    coeffs_tilde{j} = coeffs{j};
end

shuffled_coeffs = {};
for j=size(coeffs, 2):-1:size(coeffs, 2) - randomise_from_scale
    shuffled_coeffs{j} = coeffs{j};
end

% run for each desired scale
for j=size(coeffs,2) - randomise_from_scale - 1:-1:1
  
    % get multiplicators for scale j
    multiplicators = zeros(size(coeffs{j}));
    for k=1:size(coeffs{j+1}, 2)
        multiplicators(1, 2*k - 1) = coeffs{j}(1, 2*k - 1) / coeffs{j+1}(1, k);
        multiplicators(1, 2*k) = coeffs{j}(1, 2*k) / coeffs{j+1}(1, k); 
    end
    
    % shuffle multiplicators in scale j randomly
    coef = zeros(size(multiplicators));
    rnd_idx = randperm(size(multiplicators, 2));
    multiplicators = multiplicators(1, rnd_idx);
    
    % get coefficients with tilde according to a cascade
    for k=1:size(coeffs{j+1}, 2)
        coef(1, 2*k - 1) = multiplicators(1, 2*k - 1) * coeffs_tilde{j+1}(1, k);
        coef(1, 2*k) = multiplicators(1, 2*k) * coeffs_tilde{j+1}(1, k);
       coeffs_tilde{j} = coef;
    end
    
    % sort original coefficients
    coeffs{j} = sort(coeffs{j});
    
    % sort args for shuffled coefficients
    [t, idx] = sort(coeffs_tilde{j});
    
    % finally, rearange original coefficient according to coefficient with tilde
    shuffled_coeffs{j} = coeffs{j}(1, idx);
    
end

surr = idwt(shuffled_coeffs, n-1, sf);

%%
% if amplitude adjust surrogates
if amplitude_adjust_surrogates
    
    % sort generated surrogates
    [t, idx] = sort(surr);
    
    % amplitude adjusted surrogates are original data sorted according to the surrogates
    ts = sort(ts);
    surr = zeros(size(ts));
    surr(1, idx) = ts;
    
end