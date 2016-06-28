import numpy as np


class m_ssa():
    """
    Holds data and performs M-SSA.
    Can perform rotated M-SSA.
    """

    def __init__(self, X, M):
        """
        X is input data matrix as time x dimension -- N x D.
        If X is univariate, analysis could be performed as well, M-SSA reduces to classic SSA.
        M is embedding window. 
        """

        if X.ndim == 1:
            X = np.atleast_2d(X).T
        self.X = X
        self.n, self.d = X.shape
        self.M = M
        self.T = None



    def _shift(self, arr, n, order = 'forward'):
        """
        Helper function for time embedding. 
        """

        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        if order == 'forward':
            shifted = arr[n:] + [0] * n
        elif order == 'reversed':
            shifted = [0] * n + arr[:-n]
        else:
            print("Order %s not recognized.  Try forward or reversed" % order)

        return shifted



    def run_ssa(self):
        """
        Performs multichannel SSA (M-SSA) on data matrix X.
        X as time x dimension -- N x D. 
        For simplicity of computing reconstructed components, the 
        PC time series have full length of N, not N-M+1.

        According to Groth & Ghil (2011), Physical Review E, 84(3).

        Return as eigenvalues (M*D), eigenvectors (M*D x M*D), 
        principal components (N x M*D) and recontructed components (N x D x M).
        """

        # center and normalise
        for i in range(self.d):
            self.X[:, i] -= np.mean(self.X[:, i])
            self.X[:, i] /= np.std(self.X[:, i], ddof = 1)

        # embed
        aug_x = np.zeros((self.n, self.d*self.M))
        for ch in range(self.d):
            tmp = np.c_[[self._shift(self.X[:, ch], m) for m in range(self.M)]].T
            aug_x[:, ch*self.M:(ch+1)*self.M] = tmp

        # cov matrix
        self.C = np.dot(aug_x.T, aug_x) / self.n

        # eigendecomposition
        u, lam, e = np.linalg.svd(self.C, compute_uv = True) # diag(lambda) = E.T * C * E, lambda are eigenvalues, cols of E are eigenvectors
        e = e.T
        assert np.allclose(np.diag(lam), np.dot(e.T, np.dot(self.C, e)))
        
        self.lambda_sum = np.sum(lam)
        lam /= self.lambda_sum
        ndx = lam.argsort()[::-1]
        self.e = e[:, ndx] # d*M x d*M
        self.lam = lam[ndx]

        # principal components
        self.pc = np.dot(aug_x, e)

        # reconstructed components
        # self.rc = np.zeros((self.n, self.d, self.M))
        self.rc = np.zeros((self.n, self.d, self.d*self.M))
        for ch in range(self.d):
            for m in np.arange(self.d*self.M):
                Z = np.zeros((self.n, self.M))  # Time-delayed embedding of PC[:, m].
                for m2 in np.arange(self.M):
                    Z[m2 - self.n:, m2] = self.pc[:self.n - m2, m]

                # Determine RC as a scalar product.
                self.rc[:, ch, m] = np.dot(Z, self.e[ch*self.M:(ch+1)*self.M, m] / self.M)


        return self.lam, self.e, self.pc, np.squeeze(self.rc)



    def _get_structured_varimax_rotation_matrix(self, gamma = 1., q = 20, tol = 1e-6):
        """
        Computes the rotation matrix T.
        S is number of eigenvectors entering the rotation

        Adapted from Portes & Aguirre (2016), Physical Review E, 93(5).

        Returns rotation matrix T.
        """

        Ascaled = (self.lam[:self.S]**2) * self.e[:, :self.S]

        p, k = Ascaled.shape
        T, d = np.eye(k), 0

        vec_i = np.array(self.M*[1]).reshape((1, self.M))
        I_d = np.eye(self.d)
        I_d_md = np.kron(I_d, vec_i)
        M = I_d - (gamma/self.d) * np.ones((self.d, self.d))
        IMI = np.dot(I_d_md.T, np.dot(M, I_d_md))

        for i in range(q):
            d_old = d
            B = np.dot(Ascaled, T)
            G = np.dot(Ascaled.T, B * np.dot(IMI, B**2))
            u, s, vh = np.linalg.svd(G)
            T = np.dot(u, vh)
            d = sum(s)
            if d_old != 0 and d/d_old < 1 + tol:
                break

        # T is rotation matrix
        self.T = T



    def _get_orthomax_rotation_matrix(self, gamma = 1.0, q = 20, tol = 1e-6):
        """
        Computes the rotation matrix T.
        S is number of eigenvectors entering the rotation

        Adapted from Portes & Aguirre (2016), Physical Review E, 93(5).

        Returns rotation matrix T.
        """

        Ascaled = (self.lam[:self.S]**2) * self.e[:, :self.S]

        p, k = Ascaled.shape
        R, d = np.eye(k), 0

        for i in range(q):
            d_old = d
            Lambda = np.dot(Ascaled, R)
            u, s, vh = np.linalg.svd(np.dot(Ascaled.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
            R = np.dot(u,vh)
            d = np.sum(s)
            if d_old != 0 and d/d_old < 1 + tol: 
                break
        
        # R is rotation matrix
        self.T = R



    def apply_varimax(self, S, structured = True, sort_lam = False):
        """
        Performs varimax rotation on M-SSA eigenvectors.
        S is number of eigenvectors entering the rotation.
        If structured is True, applies structured varimax rotation, if False applies basic orthomax.

        Returns as M-SSA, but rotated.
        """

        self.S = S
        if structured:
            self._get_structured_varimax_rotation_matrix()
        else:
            self._get_orthomax_rotation_matrix()

        # rotated eigenvectors
        self.Es_rot = np.dot(self.e[:, :self.S], self.T)

        # rotated eigenvalues
        m_lam = np.diag(self.lam[:self.S])
        self.lam_rot = np.diag(np.dot(self.T.T, np.dot(m_lam, self.T)))
        if sort_lam:
            ndx = self.lam_rot.argsort()[::-1]
            self.lam_rot = self.lam_rot[ndx]
            self.Es_rot = self.Es_rot[:, ndx]

        # rotated PCs
        self.pc_rot = np.dot(self.pc[:, :self.S], self.T)

        # rotated RCs
        self.rc_rot = np.zeros((self.n, self.d, self.d*self.M))
        pc_mix = self.pc.copy()
        pc_mix[:, :self.S] = self.pc_rot.copy()
        e_mix = self.e.copy()
        e_mix[:, :self.S] = self.Es_rot.copy() 
        for ch in range(self.d):
            for m in np.arange(self.d*self.M):
                Z = np.zeros((self.n, self.M))  # Time-delayed embedding of PC[:, m].
                for m2 in np.arange(self.M):
                    Z[m2 - self.n:, m2] = pc_mix[:self.n - m2, m]

                # Determine RC as a scalar product.
                self.rc_rot[:, ch, m] = np.dot(Z, e_mix[ch*self.M:(ch+1)*self.M, m] / self.M)

        return self.lam_rot, self.Es_rot, self.pc_rot, np.squeeze(self.rc_rot)