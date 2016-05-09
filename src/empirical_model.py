from data_class import DataField
import numpy as np


def _partial_least_squares(x, y, ux, sx, vx, optimal, intercept = True):
    """
    Partial least squares for regression with regularization. 
    """

    xm = np.mean(x, axis = 0)
    ym = np.mean(y, axis = 0)

    e = ux * sx
    f = y - ym

    w = np.zeros((x.shape[1], optimal))
    if len(y.shape) == 1:
        yshp = 1
    else:
        yshp = y.shape[1]
    c = np.zeros((yshp, optimal))
    t = np.zeros((x.shape[0], optimal))
    u = np.zeros_like(t)
    b = np.zeros((optimal, optimal))
    p = np.zeros_like(w)

    for j in range(optimal):
        l = np.dot(e.T, f)
        if len(l.shape) == 1:
            l = l[:, np.newaxis]
        rv, _, lv = np.linalg.svd(l, full_matrices = False)
        rv, lv = rv[:, 0], np.squeeze(lv.T[:, 0])

        w[:, j] = rv
        c[:, j] = lv
        t[:, j] = np.dot(e, w[:, j])
        t[:, j] /= np.sqrt(np.dot(t[:, j].T, t[:, j]))
        u[:, j] = np.dot(f, np.squeeze(c[:, j]))
        b[j, j] = np.dot(t[:, j].T, u[:, j])
        p[:, j] = np.dot(e.T, t[:, j])

        e -= np.outer(t[:, j], p[:, j].T)
        f -= np.squeeze(np.dot(b[j, j], np.outer(t[:, j], np.squeeze(c[:, j].T))))

    bpls1 = np.dot(np.dot(np.linalg.pinv(p[:, :optimal].T), b[:optimal, :optimal]), np.squeeze(c[:, :optimal].T))
    bpls2 = np.dot(vx[:, :sx.shape[0]], bpls1)

    if intercept:
        # bpls = np.zeros((bpls2.shape[0] + 1, bpls2.shape[1]))
        # bpls[:-1, :] = bpls2
        # bpls[-1, :] = ym - np.dot(xm,bpls2)
        bpls = np.append(bpls2, [ym - np.dot(xm,bpls2)])
        xx = np.c_[ x, np.ones(x.shape[0]) ]
        r = y - np.dot(xx, bpls)
    else:
        bpls = bpls2
        r = y - np.dot(x, bpls)

    return bpls, r


def cross_correlation(a, b, max_lag):
    """
    Cross correlation with lag.
    """

    a = (a - np.mean(a)) / (np.std(a, ddof = 1) * (len(a) - 1))
    b = (b - np.mean(b)) / np.std(b, ddof = 1)
    cor = np.correlate(a, b, 'full')

    return cor[len(cor)//2 - max_lag : len(cor)//2 + max_lag+1]


def kdensity_estimate(a, kernel = 'gaussian', bandwidth = 1.0):
    """
    Estimates kernel density. Uses sklearn.
    kernels: 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'
    """

    from sklearn.neighbors import KernelDensity
    a = a[:, None]
    x = np.linspace(a.min(), a.max(), 100)[:, None]
    kde = KernelDensity(kernel = kernel, bandwidth = bandwidth).fit(a)
    logkde = kde.score_samples(x)

    return x, np.exp(logkde)


class EmpiricalModel(DataField):
    """
    Class holds the geo data and is able to fit / train and integrate statistical model
    as in Kravtsov et al., J. Climate, 18, 4404 - 4424 (2005).
    Working with monthly data. 
    """

    def __init__(self, no_levels, verbose = False):
        """
        Init function.
        """

        DataField.__init__(self)
        self.no_levels = no_levels
        self.low_freq = None
        self.input_pcs = None
        self.input_eofs = None
        self.verbose = verbose



    def load_geo_data(self, fname, varname, start_date, end_date, lats = None, lons = None, dataset = 'NCEP', anom = False):
        """
        Loads geo data, makes sure the data is monthly.
        """

        self.load(fname, varname, dataset = dataset, print_prog = False)
        self.select_date(start_date, end_date)
        self.select_lat_lon(lats, lons)
        if anom:
            self.anomalise()
        if np.abs(self.time[1] - self.time[0]) <= 1:
            raise Exception("Model works only with monthly data.")
        if self.verbose:
            print("Data loaded with shape %s and time span %s -- %s." 
                % (self.data.shape, self.get_date_from_ndx(0), self.get_date_from_ndx(-1)))



    def copy_existing_datafield(self, g):
        """
        Copies existing DataField instance to this model.
        """

        self.data = g.data.copy()
        self.time = g.time.copy()
        if np.abs(self.time[1] - self.time[0]) <= 1:
            raise Exception("Model now works only with monthly data.")
        self.lats = g.lats.copy()
        self.lons = g.lons.copy()
        self.nans = g.nans
        if self.verbose:
            print("DataField copied to model. Shape of the data is %s, time span is %s -- %s including." 
                % (self.data.shape, self.get_date_from_ndx(0), self.get_date_from_ndx(-1)))



    def remove_low_freq_variability(self, mean_over, cos_weights = True, no_comps = None):
        """
        Removes low-frequency variability (usually magnitude of decades) and 
        stores the signal in EOFs.
        mean_over in years, cos_weights whether to use cosine reweighting
        if no_comps is None, keeps number such 99% of variance is described.
        """

        if self.verbose:
            print("removing low frequency variability...")

        window = int((mean_over / 2.) * 12.)

        # boxcar mean
        smoothed = np.zeros_like(self.data)
        if self.verbose:
            print("...running boxcar mean over %d years..." % mean_over)
        for t in range(self.time.shape[0]):
            smoothed[t, ...] = np.nanmean(self.data[max(t-window,0) : min(t+window, self.time.shape[0]), ...], axis = 0)

        # cos-weighting
        if cos_weights:
            if self.verbose:
                print("...scaling by square root of cosine of latitude...")
            cos_w = self.latitude_cos_weights()
            smoothed *= cos_w

        # pca on low-freq field
        if no_comps is not None:
            if self.verbose:
                print("...storing low frequency variability in %d EOFs..." % no_comps)
            eofs, pcs, var, pca_mean = self.pca_components(n_comps = no_comps, field = smoothed)
            if self.verbose:
                print("...which explain %.2f%% of total low frequency variability..." % (np.sum(var)*100.))
        elif no_comps is None:
            if self.verbose:
                print("...storing low frequency variability in EOFs such that they explain 99% of variability...")
            eofs, pcs, var, pca_mean = self.pca_components(n_comps = 20, field = smoothed)
            idx = np.where(np.cumsum(var) > 0.99)[0][0] + 1
            eofs, pcs, var = eofs[:idx, ...], pcs[:idx, ...], var[:idx]
        self.low_freq = [eofs, pcs, var, pca_mean]

        # subtract from data
        if self.verbose:
            print("...subtracting from data...")
        if self.nans:
            self.data = self.filter_out_NaNs()[0]
            self.data -= pca_mean
            self.data = self.return_NaNs_to_data(self.data)
        else:
            self.flatten_field()
            self.data -= pca_mean
            self.reshape_flat_field()
        temp = self.flatten_field(eofs)
        self.flatten_field()
        self.data -= np.dot(temp.T, pcs).T
        self.reshape_flat_field()
        if self.verbose:
            print("done.")



    def prepare_input(self, anom = True, no_input_ts = 20, cos_weights = True, sel = None):
        """
        Prepares input time series to model as PCs.
        if sel is not None, selects those PCs as input (sel pythonic, starting with 0).
        """

        if self.verbose:
            print("preparing input to the model...")

        if anom:
            if self.verbose:
                print("...anomalising...")
            self.anomalise()

        if cos_weights:
            if self.verbose:
                print("...scaling by square root of cosine of latitude...")
            cos_w = self.latitude_cos_weights()
            self.data *= cos_w

        if sel is None:
            if self.verbose:
                print("...selecting %d first principal components as input time series..." % (no_input_ts))
            eofs, pcs, var = self.pca_components(no_input_ts)
            self.input_pcs = pcs
            self.input_eofs = eofs
            if self.verbose:
                print("...and they explain %.2f%% of variability..." % (np.sum(var)*100.))
        else:
            if self.verbose:
                print("...selecting %d principal components described in 'sel' variable..." % (len(sel)))
            eofs, pcs, var = self.pca_components(sel[-1]+1)
            self.input_pcs = pcs[sel, :]
            self.input_eofs = eofs[sel, ...]
            if self.verbose:
                print("...and they explain %.2f%% of variability..." % (np.sum(var[sel])*100.))
        if self.verbose:
            print("done.")


    def train_model(self, harmonic_pred = 'first', quad = False):
        """
        Train the model.
        harmonic_pred could have values 'first', 'all', 'none'
        if quad, train quadratic model, else linear
        """

        self.harmonic_pred = harmonic_pred
        self.quad = quad

        if self.verbose:
            print("now training %d-level model..." % self.no_levels)
        # standartise PCs
        pcs = self.input_pcs / np.std(self.input_pcs[0, :], ddof = 1)

        if harmonic_pred not in ['first', 'none', 'all']:
            raise Exception("Unknown keyword for harmonic predictor, please use: 'first', 'all' or 'none'.")

        if harmonic_pred in ['all', 'first']:
            if self.verbose:
                print("...using harmonic predictors (with annual frequency)...")
            xsin = np.sin(2*np.pi*np.arange(pcs.shape[1]) / 12.)
            xcos = np.cos(2*np.pi*np.arange(pcs.shape[1]) / 12.)
            

        if quad:
            if self.verbose:
                print("...training quadratic model...")

        pcs = pcs.T # time x dim

        residuals = {}
        fit_mat = {}

        for level in range(self.no_levels):

            if self.verbose:
                print("...training %d. out of %d levels..." % (level+1, self.no_levels))

            fit_mat_size = pcs.shape[1]*(level+1) + 1 # as extended vector + intercept
            if level == 0:
                if harmonic_pred in ['first', 'all']:
                    fit_mat_size += 2*pcs.shape[1] + 2 # harm
                if quad and level == 0:
                    fit_mat_size += (pcs.shape[1] * (pcs.shape[1] - 1)) / 2 # quad
            elif level > 0:
                if harmonic_pred == ['all']:
                    fit_mat_size += (level+1)*2*pcs.shape[1] + 2

            # response variables -- y (dx/dt)
            if self.verbose:
                print("...preparing response variables...")
            y = np.zeros_like(pcs)
            if level == 0:
                y[:-1, :] = np.diff(pcs, axis = 0)
            else:
                y[:-1, :] = np.diff(residuals[level-1], axis = 0)
            y[-1, :] = y[-2, :]

            fit_mat[level] = np.zeros((fit_mat_size, pcs.shape[1]))
            residuals[level] = np.zeros_like(pcs)

            for k in range(pcs.shape[1]):
                # prepare predictor
                x = pcs.copy()
                for l in range(level):
                    x = np.c_[x, residuals[l]]
                if level == 0:
                    if quad:
                        quad_pred = np.zeros((pcs.shape[0], (pcs.shape[1]*(pcs.shape[1] - 1))/2))
                        for t in range(pcs.shape[0]):
                            q = np.tril(np.outer(pcs[t, :].T, pcs[t, :]), -1)
                            quad_pred[t, :] = q[np.nonzero(q)]
                    if harmonic_pred in ['all', 'first']:
                        if quad:
                            x = np.c_[quad_pred, x, x*np.outer(xsin,np.ones(x.shape[1])), x*np.outer(xcos,np.ones(x.shape[1])), 
                                xsin, xcos]
                        else:
                            x = np.c_[x, x*np.outer(xsin,np.ones(x.shape[1])), x*np.outer(xcos,np.ones(x.shape[1])), 
                                xsin, xcos]
                    else:
                        if quad:
                            x = np.c_[quad_pred, x]
                else:
                    if harmonic_pred == ['all']:
                        x = np.c_[x, x*np.outer(xsin,np.ones(x.shape[1])), x*np.outer(xcos,np.ones(x.shape[1])), 
                                xsin, xcos]

                # regularize and regress
                x -= np.mean(x, axis = 0)
                ux,sx,vx = np.linalg.svd(x, False)
                optimal = min(ux.shape[1], 25)
                b_aux, residuals[level][:, k] = _partial_least_squares(x, y[:, k], ux, sx, vx.T, optimal, True)

                # store results
                fit_mat[level][:, k] = b_aux

                if (k+1)%10==0 and self.verbose:
                    print("...%d/%d finished fitting..." % (k+1, pcs.shape[1]))

            if self.verbose:
                # finish check for negative definiteness
                d, _ = np.linalg.eig(fit_mat[level][:pcs.shape[1], :pcs.shape[1]])
                print("...maximum eigenvalue: %.4f" % (max(np.real(d))))

        self.residuals = residuals
        self.fit_mat = fit_mat

        if self.verbose:
            print("training done.")



    def integrate_model(self, n_realizations, int_length = None, noise_type = ['white'], sigma = 1., n_workers = 3, diagnostics = True):
        """
        Integrate trained model.
        noise_type:
        -- white - classic white noise, spatial correlation by cov. matrix of last level residuals
        ---- ['white']
        -- cond - find n_samples closest to the current space in subset of n_pcs and use their cov. matrix
        ---- ['cond', n_samples, n_pcs]
        -- seasonal - seasonal dependence of the residuals, fit n_harm harmonics of annual cycle, could also be used with cond.
        ---- ['seasonal', n_harmonics, True/False]
        -- 'extended' - uses extended cov. matrix with lags with n_snippet max-lag, could also be used with seasonal
        ---- ['extended', n_snippet, n_pcs, True/False]
        """

        if self.verbose:
            print("preparing to integrate model...")
        # standartise PCs
        pcs = self.input_pcs / np.std(self.input_pcs[0, :], ddof = 1)
        pcs = pcs.T # time x dim

        pcmax = np.amax(pcs, axis = 0)
        pcmin = np.amin(pcs, axis = 0)
        varpc = np.var(pcs, axis = 0, ddof = 1)
        
        self.int_length = pcs.shape[0] if int_length is None else int_length

        if self.harmonic_pred in ['all', 'first']:
            if self.verbose:
                print("...using harmonic predictors (with annual frequency)...")
            self.xsin = np.sin(2*np.pi*np.arange(pcs.shape[1]) / 12.)
            self.xcos = np.cos(2*np.pi*np.arange(pcs.shape[1]) / 12.)

        if noise_type[0] not in ['white', 'cond', 'seasonal', 'extended']:
            raise Exception("Unknown noise type to be used as forcing. Use 'white', 'cond', 'seasonal' or 'extended'.")
        
        if noise_type[0] == 'white':
            Q = np.cov(self.residuals[max(residuals.keys())], rowvar = 0)
            self.rr = np.linalg.cholesky(Q).T

        if diagnostics:
            if self.verbose:
                print("...running diagnostics for the data...")
            # ACF, kernel density, integral corr. timescale for data
            max_lag = 50
            lag_cors = np.zeros((2*max_lag + 1, pcs.shape[1]))
            kernel_densities = np.zeros((100, pcs.shape[1], 2))
            for k in range(pcs.shape[1]):
                lag_cors[:, k] = cross_correlation(pcs[:, k], pcs[:, k], max_lag = max_lag)
                kernel_densities[:, k, 0], kernel_densities[:, k, 1] = kdensity_estimate(pcs[:, k], kernel = 'epanechnikov')
            integral_corr_timescale = np.sum(np.abs(lag_cors), axis = 0)

            # init for integrations
            lag_cors_int = np.zeros([n_realizations] + list(lag_cors.shape))
            kernel_densities_int = np.zeros([n_realizations] + list(kernel_densities.shape))
            stat_moments_int = np.zeros((4, n_realizations, pcs.shape[1])) # mean, variance, skewness, kurtosis

        self.diagpc = np.diag(np.std(pcs, axis = 0, ddof = 1))
        self.maxpc = np.amax(np.abs(pcs))
        self.diagres = {}
        self.maxres = {}
        for l in self.residuals.keys():
            self.diagres[l] = np.diag(np.std(self.residuals[l], axis = 0, ddof = 1))
            self.maxres[l] = np.amax(np.abs(self.residuals[l]))

        if n_workers > 1:
            from multiprocessing import Pool
            pool = Pool(n_workers)
            map_func = pool.map
            if self.verbose:
                print("...starting integration of %d realizations using %d workers..." % (n_realizations, n_workers))
        else:
            map_func = map
            if self.verbose:
                print("...starting integration of %d realizations using single thread..." % n_realizations)

        args = []


    def _process_integration(self, rnd):

        num_exploding = 0
        repeats = 20
        xx = {}
        for l,r in zip(self.fit_mat.keys(), rnd):
            xx[l] = np.zeros((repeats, self.input_pcs.shape[0]))
            xx[l][0, :] = r

            x[l] = np.zeros((self.int_length, self.input_pcs.shape[0]))
            x[l][0, :] = xx[l][0, :]

        step0 = 0
        step = 1
        for n in range(repeats*np.ceil(self.int_length/repeats)):
            for k in range(2, repeats):
                # zz = 
                pass

        