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


	def get_phase_coherence_matrix(self):
		"""
		Gets the matrix of mean phase coherence between each two grid-points.
		"""



