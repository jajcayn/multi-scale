import numpy as np
from src.data_class import load_NCEP_data_monthly, load_NCEP_data_daily
import src.wavelet_analysis as wvlt

class ScaleSpecificNetwork():
	"""
	Class holds geo data and can construct networks.
	"""

	def __init__(self, fname, varname, start_date, end_date, sampling = 'monthly', anom = False):
		"""
		Initialisation of the class.
		"""

		if sampling == 'monthly':
			self.g = load_NCEP_data_monthly(fname, varname, start_date, end_date, None, None, None, anom)
		elif sampling == 'daily':
			self.g = load_NCEP_data_daily(fname, varname, start_date, end_date, None, None, None, anom)

		self.phase = None
		self.amplitude = None
		self.coherence_matrix = None

		self.num_lats = self.g.lats.shape[0]
		self.num_lons = self.g.lons.shape[0]


	def _get_oscillatory_modes(a):
	    """
	    Gets oscillatory modes in terms of phase and amplitude from wavelet analysis for given data.
	    """
	    i, j, s0, data = a
	    wave, _, _, _ = wvlt.continous_wavelet(data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
	    phase = np.arctan2(np.imag(wave), np.real(wave))
	    amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
	    
	    return i, j, phase, amplitude


	def wavelet(self, period, pool = None):
		"""
		Performs wavelet analysis on the data.
		"""

		k0 = 6. # wavenumber of Morlet wavelet used in analysis, suppose Morlet mother wavelet
		y = 365.25 # year in days, for periods at least 4years totally sufficient, effectively omitting leap years
		fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
		per = period * y # frequency of interest
		s0 = per / fourier_factor # get scale

		self.phase = np.zeros_like(self.g.data)
		self.amplitude = np.zeros_like(self.g.data)

		if pool is None:
			map_func = map
		elif pool is not None:
			map_func = pool.map

		job_args = [ (i, j, s0, self.g.data[:, i, j]) for i in range(self.num_lats) for j in range(self.num_lons) ]
		job_result = map_func(self._get_oscillatory_modes, job_args)
		del job_args

		for i, j, ph, am in job_result:
			self.phase[:, i, j] = ph
			self.amplitude[:, i, j] = am

		del job_result


	def _get_phase_coherence(a):
		"""
		Gets mean phase coherence for given data.
		"""

		i, j, ph1, ph2 = a

		# get continuous phase
		for ii in range(ph1.shape[0] - 1):
			if np.abs(ph1[ii+1] - ph1[ii]) > 1:
				ph1[ii+1: ] += 2 * np.pi
			if np.abs(ph2[ii+1] - ph2[ii]) > 1:
				ph2[ii+1: ] += 2 * np.pi

		# get phase diff
		diff = ph1 - ph2

		# compute mean phase coherence
		coh = np.mean(np.cos(diff)) * np.mean(np.cos(diff)) + np.mean(np.sin(diff)) * np.mean(np.sin(diff))

		return i, j, coh


	def get_phase_coherence_matrix(self, pool = None):
		"""
		Gets the matrix of mean phase coherence between each two grid-points.
		"""
		
		self.phase = self.g.flatten_field(self.phase)

		self.coherence_matrix = np.zeros((self.phase.shape[1], self.phase.shape[1]))

		if pool is None:
			map_func = map
		elif pool is not None:
			map_func = pool.map

		job_args = [ (i, j, self.phase[:, i], self.phase[:, j]) for i in range(self.phase.shape[1]) for j in range(i, self.phase.shape[1]) ]
		job_results = map_func(self._get_phase_coherence, job_args)
		del job_args

		for i, j, coh in job_results:
			self.coherence_matrix[i, j] = coh

		del job_results


