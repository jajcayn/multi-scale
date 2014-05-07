"""
created on May 6, 2014

@author: Nikola Jajcay
"""

import numpy as np
from data_class import load_station_data
import wavelet_analysis
import matplotlib.pyplot as plt
from surrogates.surrogates import SurrogateField


class OscillatoryTimeSeries:
    
    def __init__(self, fname, start_date, end_date, anom = True):
        """
        Loads data.
        """
        self.g = load_station_data(fname, start_date, end_date, anom)
        self.phase = None
        self.amplitude = None
        self.cond_means = None
        self.cond_variance = None
        self.cond_means_surrogates = None
        if anom:
            self.sat = 'SATA'
        else:
            self.sat = 'SAT'
        
        
    def wavelet(self, period):
        """
        Performs wavelet.
        """
        self.period = period
        k0 = 6. # wavenumber of Morlet wavelet used in analysis
        y = 365.25 # year in days
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
        per = period * y # frequency of interest
        s0 = per / fourier_factor # get scale
        self.scale = s0
        wave, _, _, _ = wavelet_analysis.continous_wavelet(self.g.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
        
        phase = np.arctan2(np.imag(wave), np.real(wave))
        self.phase = phase[0, :]
        amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
        self.amplitude = amplitude[0, :]
        
        
    def get_conditional_means(self, bin_no = None, means = True):
        """
        Gets conditional means (if means = True) or conditional variance (means = False).
        """
        if bin_no is None:
            bin_no = self.period
        if self.phase is not None:
            phase_bins = np.array(np.linspace(-np.pi, np.pi, bin_no+1))
            cond_means = np.zeros((bin_no,))
            for i in range(cond_means.shape[0]):
                ndx = ((self.phase >= phase_bins[i]) & (self.phase <= phase_bins[i+1]))
                if means:
                    cond_means[i] = np.mean(self.g.data[ndx])
                else:
                    cond_means[i] = np.var(self.g.data[ndx], ddof = 1)
            
            self.phase_bins = phase_bins
            if means:
                self.cond_means = cond_means
                self.cond_variance = None
            else:
                self.cond_means = None
                self.cond_variance = cond_means
            return cond_means
        
        else:
            raise Exception('First perform a wavelet transform to obtain phase!')
        
        
    def get_conditional_means_surrogates(self, num_surr, MF_surr = True):
        """
        Gets conditional means/variance for num_surr surrogates.
        If MF_surr is True, multifractal surrogates are used, otherway FT surrogates are used.
        """
        
        if self.cond_means is not None or self.cond_variance is not None:
            cond_means_surr = np.zeros((num_surr, self.phase_bins.shape[0] - 1))
            
            # get seasonality from data
            mean, var, trend = self.g.get_seasonality(True)
            self.sg = SurrogateField()
            self.sg.copy_field(self.g)
            
            for su in range(num_surr):
                if MF_surr:
                    self.sg.construct_multifractal_surrogates()
                else:
                    self.sg.construct_fourier_surrogates_spatial()
                self.sg.add_seasonality(mean, var, trend)
                wave, _, _, _ = wavelet_analysis.continous_wavelet(self.sg.surr_data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = self.scale, j1 = 0, k0 = 6.) # perform wavelet
                phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
                for i in range(cond_means_surr.shape[1]):
                    ndx = ((phase >= self.phase_bins[i]) & (phase <= self.phase_bins[i+1]))[0]
                    if self.cond_variance is None:
                        cond_means_surr[su, i] = np.mean(self.sg.surr_data[ndx])
                    elif self.cond_means is None:
                        cond_means_surr[su, i] = np.var(self.sg.surr_data[ndx], ddof = 1)
                        
                if (su+1) % 10 == 0:
                    print("%d/%d surrogates done..." % (su+1, num_surr))
                    
            self.cond_means_surrogates = cond_means_surr
            return cond_means_surr
            
        else:
            raise Exception('First obtain conditional means/variance on data, than on surrogates!')
        
        
    def plot_conditional_means(self, fname = None, plot_surrogates = False):
        """
        Plots conditional means/variance as a bar plot.
        """
        if self.cond_means is not None or self.cond_variance is not None:
            diff = (self.phase_bins[1] - self.phase_bins[0])
            fig = plt.figure(figsize=(6,10))
            if self.cond_variance is None:
                if not plot_surrogates:
                    b1 = plt.bar(self.phase_bins[:-1]+diff*0.05, self.cond_means, width = diff*0.9, bottom = None, fc = '#403A37', figure = fig)
                    plt.title("%s -- cond. means %s \n difference: %.2f$^{\circ}$C" % (self.g.location, self.sat, (self.cond_means.max() - self.cond_means.min())))
                elif plot_surrogates and self.cond_means_surrogates is not None:
                    b1 = plt.bar(self.phase_bins[:-1], self.cond_means, width = diff*0.45, bottom = None, fc = '#403A37', figure = fig)
                    b2 = plt.bar(self.phase_bins[:-1] + diff*0.5, np.mean(self.cond_means_surrogates, axis = 0), width = diff*0.45, bottom = None, fc = '#A09793', figure = fig)
                    plt.legend( (b1[0], b2[0]), ('data', 'mean of ' + str(self.cond_means_surrogates.shape[0]) + ' surr') )
                    mean_of_diffs = np.mean([self.cond_means_surrogates[i,:].max() - self.cond_means_surrogates[i,:].min() for i in range(self.cond_means_surrogates.shape[0])])
                    std_of_diffs = np.std([self.cond_means_surrogates[i,:].max() - self.cond_means_surrogates[i,:].min() for i in range(self.cond_means_surrogates.shape[0])], ddof = 1)
                    plt.title("%s -- cond. means %s \n difference data: %.2f$^{\circ}$C \n difference surrogates mean: %.2f$^{\circ}$C \n difference surrogates std: %.2f$^{\circ}$C" % (self.g.location, self.sat, (self.cond_means.max() - self.cond_means.min()), mean_of_diffs, std_of_diffs))
                    plt.subplots_adjust(top=0.87)
                else:
                    raise Exception('No surrogates created to plot! Either obtain surrogates or plot with plot_surrogates = False!')
                plt.ylabel('cond mean temperature [$^{\circ}$C]')
                plt.axis([-np.pi, np.pi, np.floor(self.cond_means.min() * 2) / 2, np.ceil(self.cond_means.max() * 2) / 2])
            elif self.cond_means is None:
                if not plot_surrogates:
                    b1 = plt.bar(self.phase_bins[:-1]+diff*0.05, self.cond_variance, width = diff*0.9, bottom = None, fc = '#403A37', figure = fig)
                    plt.title("%s -- cond. variance %s \n difference: %.2f$^{\circ}$C$^{2}$ \n %d years period" % (self.g.location, self.sat, (self.cond_variance.max() - self.cond_variance.min()), self.period))
                elif plot_surrogates and self.cond_means_surrogates is not None:
                    b1 = plt.bar(self.phase_bins[:-1], self.cond_variance, width = diff*0.45, bottom = None, fc = '#403A37', figure = fig)
                    b2 = plt.bar(self.phase_bins[:-1] + diff*0.5, np.mean(self.cond_means_surrogates, axis = 0), width = diff*0.45, bottom = None, fc = '#A09793', figure = fig)
                    plt.legend( (b1[0], b2[0]), ('data', 'mean of ' + str(self.cond_means_surrogates.shape[0]) + ' surr') )
                    mean_of_diffs = np.mean([self.cond_means_surrogates[i,:].max() - self.cond_means_surrogates[i,:].min() for i in range(self.cond_means_surrogates.shape[0])])
                    std_of_diffs = np.std([self.cond_means_surrogates[i,:].max() - self.cond_means_surrogates[i,:].min() for i in range(self.cond_means_surrogates.shape[0])], ddof = 1)
                    plt.title("%s -- cond. variance %s \n difference data: %.2f$^{\circ}$C$^{2}$ \n difference surrogates mean: %.2f$^{\circ}$C$^{2}$ \n difference surrogates std: %.2f$^{\circ}$C$^{2}$" % (self.g.location, self.sat, (self.cond_variance.max() - self.cond_variance.min()), mean_of_diffs, std_of_diffs))
                    plt.subplots_adjust(top=0.87)
                else:
                    raise Exception('No surrogates created to plot! Either obtain surrogates or plot with plot_surrogates = False!')
                plt.ylabel('cond variance temperature [$^{\circ}$C$^{2}$]')
                plt.axis([-np.pi, np.pi, np.floor(self.cond_variance.min() * 2) / 2, np.ceil(self.cond_variance.max() * 2) / 2])
            plt.xlabel('phase [rad]')
            if fname is None:
                plt.show()
            else:
                plt.savefig(fname)
                
        else:
            raise Exception('Nothing to plot yet! First obtain conditional means/variance!')
        
