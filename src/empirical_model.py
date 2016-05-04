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
    c = np.zeros_like(w)
    t = np.zeros((x.shape[0], optimal))
    u = np.zeros_like(t)
    b = np.zeros((optimal, optimal))
    p = np.zeros_like(w)

    for j in range(optimal):
        rv, s, lv = np.linalg.svd(np.dot(e.T, f), full_matrices = False)
        rv, s, lv = rv[:1], s[:1], lv.T[:1]

        w[:, j] = rv
        c[:, j] = lv
        t[:, j] = np.dot(e, w[:, j])
        t[:, j] /= np.sqrt(np.dot(t[:, j].T, t[:, j]))
        u[:, j] = np.dot(f, c[:, j])
        b[j, j] = np.dot(t[:, j].T, u[:, j])
        p[:, j] = np.dot(e.T, t[:, j])

        e -= np.outer(t[:, j], p[:, j].T)
        f -= np.dot(b[j, j], np.outer(t[:, j], c[:, j].T))

    bpls1 = np.dot(np.dot(np.linalg.pinv(p[:, :optimal].T), b[:optimal, :optimal]), c[:, :optimal].T)
    bpls2 = np.dot(vx[:, :sx.shape[0]], bpls1)

    if intercept:
        bpls = np.zeros((bpls2.shape[0] + 1, bpls2.shape[1]))
        bpls[:-1, :] = bpls2
        bpls[-1, :] = ym - np.dot(xm,bpls2)
        xx = np.c_[ x, np.ones(x.shape[0]) ]
        r = y - np.dot(xx, bpls)
    else:
        bpls = bpls2
        r = y - np.dot(x, bpls)

    return bpls, r





class EmpiricalModel(DataField):
    """
    Class holds the geo data and is able to fit / train and integrate statistical model
    as in Kravtsov et al., J. Climate, 18, 4404 - 4424 (2005).
    Working with monthly data. 
    """

    def __init__(self, no_levels):
        """
        Init function.
        """

        DataField.__init__(self)
        self.no_levels = no_levels
        self.low_freq = None
        self.input_pcs = None
        self.input_eofs = None



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



    def copy_existing_datafield(self, g):
        """
        Copies existing DataField instance to this model.
        """

        self.data = g.data.copy()
        self.time = g.time.copy()
        self.lats = g.lats.copy()
        self.lons = g.lons.copy()
        self.nans = g.nans
        if np.abs(self.time[1] - self.time[0]) <= 1:
            raise Exception("Model works only with monthly data.")



    def remove_low_freq_variability(self, mean_over, cos_weights = True, no_comps = None):
        """
        Removes low-frequency variability (usually magnitude of decades) and 
        stores the signal in EOFs.
        mean_over in years, cos_weights whether to use cosine reweighting
        if no_comps is None, keeps number such 99% of variance is described.
        """

        window = int((mean_over / 2.) * 12.)

        # boxcar mean
        smoothed = np.zeros_like(self.data)
        for t in range(self.time.shape[0]):
            smoothed[t, ...] = np.nanmean(self.data[max(t-window,0) : min(t+window, self.time.shape[0]), ...], axis = 0)

        # cos-weighting
        if cos_weights:
            cos_w = self.latitude_cos_weights()
            smoothed *= cos_w

        # pca on low-freq field
        if no_comps is not None:
            eofs, pcs, var, pca_mean = self.pca_components(n_comps = no_comps, field = smoothed)
        elif no_comps is None:
            eofs, pcs, var, pca_mean = self.pca_components(n_comps = 20, field = smoothed)
            idx = np.where(np.cumsum(var) > 0.99)[0][0] + 1
            eofs, pcs, var = eofs[:idx, ...], pcs[:idx, ...], var[:idx]
        self.low_freq = [eofs, pcs, var, pca_mean]

        # subtract from data
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



    def prepare_input(self, no_input_ts = 20, cos_weights = True, sel = None):
        """
        Prepares input time series to model as PCs.
        """

        if cos_weights:
            cos_w = self.latitude_cos_weights()
            self.data *= cos_w

        eofs, pcs, var = self.pca_components(no_input_ts)
        if sel is None:
            self.input_pcs = pcs
            self.input_eofs = eofs



    def train_model(self, harmonic_pred = True, quad = False):
        """
        Train the model.
        if harmonic_pred, use annual harmonics
        if quad, train quadratic model, else linear
        """

        # standartise PCs
        pcs = self.input_pcs / np.std(self.input_pcs[0, :], ddof = 1)

        fit_mat_size = pcs.shape[0] + 1

        if harmonic_pred:
            xsin = np.sin(2*np.pi*np.arange(pcs.shape[1]) / 12.)
            xcos = np.cos(2*np.pi*np.arange(pcs.shape[1]) / 12.)
            fit_mat_size += 2*pcs.shape[0] + 2

        if quad:
            fit_mat_size += (pcs.shape[0] * (pcs.shape[0] - 1)) / 2

        # response variables -- y (dx/dt)
        y = np.zeros_like(pcs)
        y[:, :-1] = np.diff(pcs, axis = 1)
        y[:, -1] = y[:, -2]

        fit_mat = np.zeros((fit_mat_size, pcs.shape[0]))



        










