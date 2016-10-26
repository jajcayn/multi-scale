import numpy as np
from scipy import special, spatial
from mutual_information import knn_cond_mutual_information


class Chain_of_Rosslers():

    def __init__(self, J, omega_1 = 1., alpha = 0.15, coupling = 0):
        self.J = J
        self.omega_1 = omega_1
        self.alpha = alpha
        self.coupling = coupling

    def _rossler_deriv(self, Y, t0):
        result = []
        for j in range(self.J):
            x, y, z = Y[j*3], Y[j*3+1], Y[j*3+2]
            y_minus = Y[(j-1)*3+1] if j > 0 else Y[-2]
            y_plus = Y[(j+1)*3+1] if j < (self.J-1) else Y[1]
            omega_j = self.omega_1 + 0.02*j
            xres = -omega_j*y - z
            yres = omega_j*x + self.alpha*y + self.coupling*(y_plus - 2*y + y_minus)
            zres = 0.1 + z*(x - 8.5)
            result += [xres, yres, zres]

        return result

    def integrate(self, N, dt_sam = 0.4, dt = 0.01):
        import scipy.integrate as sint

        t = np.arange(0, N*dt_sam, dt)
        y0 = np.random.rand(3*self.J)
        sol = sint.odeint(self._rossler_deriv, y0, t)

        self.solution_full = sol.copy()

        # subsample
        sol = sol[::dt_sam//dt]

        return sol


def _estimate_cmi_knn(array, k, xyz, standardize=True,
                      verbosity=0):
    """Returns CMI estimate as described in Frenzel and Pompe PRL (2007).
    Args:
        array (array, optional): Data array of shape (dim, T).
        xyz (array): XYZ identifier array of shape (dim,).
        standardize (bool, optional): Whether to standardize data before.
        k (int): Number of nearest neighbors in joint space.
        verbosity (int, optional): Level of verbosity.
    Returns:
        TYPE: Description
    """
    k_xz, k_yz, k_z = _get_nearest_neighbors(array=array, xyz=xyz,
                                             k=k, standardize=standardize)

    ixy_z = special.digamma(k) - (special.digamma(k_xz) +
                                  special.digamma(k_yz) -
                                  special.digamma(k_z)).mean()

    return ixy_z



def _get_nearest_neighbors(array, xyz, k, standardize=True):
    """Returns nearest neighbors according to Frenzel and Pompe (2007).
    Retrieves the distances eps to the k-th nearest neighbors for every sample
    in joint space XYZ and returns the numbers of nearest neighbors within eps
    in subspaces Z, XZ, YZ.
    Args:
        array (array, optional): Data array of shape (dim, T).
        xyz (array): XYZ identifier array of shape (dim,).
        k (int): Number of nearest neighbors in joint space.
        standardize (bool, optional): Whether to standardize data before.
    Returns:
        Tuple of nearest neighbor arrays for X, Y, and Z.
    Raises:
        ValueError: Description
    """

    # Import cython code
    try:
        import pyximport; pyximport.install()
        import tigramite_cython_code
    except ImportError:
        raise ImportError("Could not import tigramite_cython_code, please"
                          " compile cython code first as described in Readme.")

    dim, T = array.shape

    if standardize:
        # Standardize
        array = array.astype('float')
        array -= array.mean(axis=1).reshape(dim, 1)
        array /= array.std(axis=1).reshape(dim, 1)
        # FIXME: If the time series is constant, return nan rather than raising
        # Exception
        if np.isnan(array).sum() != 0:
            raise ValueError("nans after standardizing, "
                             "possibly constant array!")

    # Add noise to destroy ties...
    array += (1E-6 * array.std(axis=1).reshape(dim, 1)
              * np.random.rand(array.shape[0], array.shape[1]))

    # Use cKDTree to get distances eps to the k-th nearest neighbors for every sample
    # in joint space XYZ with maximum norm
    tree_xyz = spatial.cKDTree(array.T)
    epsarray = tree_xyz.query(array.T, k=k+1, p=np.inf, eps=0.)[0][:,k].astype('float')

    # Prepare for fast cython access

    dim_x = int(np.where(xyz == 0)[0][-1] + 1)
    dim_y = int(np.where(xyz == 1)[0][-1] + 1 - dim_x)

    k_xz, k_yz, k_z = tigramite_cython_code._get_neighbors_within_eps_cython(array, T, dim_x, dim_y, epsarray,
            k, dim)

    return k_xz, k_yz, k_z



# ros = Chain_of_Rosslers(J = 10, coupling = 0.7)
# a = ros.integrate(N = 25000)
# print a.shape


# ja = []
# jak = []

# for k in range(2,20):
#   print k
#   ja.append(knn_cond_mutual_information(a[:, 0], a[:, 1], [a[:, 2], a[:, 3], a[:, 4]], k = k, standardize = True, dualtree = True))
#   jak.append(_estimate_cmi_knn(a[:, :5], k = k, xyz = np.array([0, 1, 2, 2, 2]), standardize = True))

# import matplotlib.pyplot as plt
# plt.plot(ja, label = "ja")
# plt.plot(jak, label = "jakob")
# plt.legend()
# plt.show()