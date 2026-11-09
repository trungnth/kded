"""
Gaussian mixture test densities in 3d with exact pdf, sampling and exact
ISE of a Gaussian KDE (closed form, no grid error).
"""

import numpy as np
from numpy import pi as PI


class GaussianMixture:

    def __init__(self, name, weights, means, covs):
        self.name = name
        self.w = np.asarray(weights, dtype=float) / np.sum(weights)
        self.mu = np.atleast_2d(np.asarray(means, dtype=float))
        self.S = np.asarray(covs, dtype=float)
        self.d = self.mu.shape[1]

    def sample(self, N, rng):
        comp = rng.choice(len(self.w), size=N, p=self.w)
        out = np.empty((N, self.d))
        for m in range(len(self.w)):
            idx = comp == m
            out[idx] = rng.multivariate_normal(self.mu[m], self.S[m], idx.sum())
        return out

    def pdf(self, x):
        x = np.atleast_2d(x)
        out = np.zeros(len(x))
        for (w, mu, S) in zip(self.w, self.mu, self.S):
            out += w * _gauss(x - mu, S)
        return out

    def int_f2(self):
        """int f^2, closed form."""
        tot = 0.0
        for (w1, m1, S1) in zip(self.w, self.mu, self.S):
            for (w2, m2, S2) in zip(self.w, self.mu, self.S):
                tot += w1*w2*_gauss((m1 - m2)[None], S1 + S2)[0]
        return tot

    def ise(self, data, H, lambdas=None, chunk=2**23):
        """Exact ISE of the Gaussian KDE with kernel covariance H, or
        lambda_i^2 H for point i if `lambdas` is given (adaptive KDE):
        int (fhat - f)^2 = int fhat^2 - 2 int fhat f + int f^2."""
        N, d = data.shape
        lam2 = np.ones(N) if lambdas is None else np.asarray(lambdas)**2
        # int fhat^2 = 1/N^2 sum_ij phi_{(l_i^2 + l_j^2) H}(X_i - X_j)
        L = np.linalg.cholesky(np.linalg.inv(H))
        Y = data @ L
        sq = (Y**2).sum(1)
        norm = 1/np.sqrt((2*PI)**d * np.linalg.det(H))
        step = max(1, chunk // N)
        s = 0.0
        for i in range(0, N, step):
            D2 = sq[i:i+step, None] + sq[None, :] - 2*Y[i:i+step] @ Y.T
            V = lam2[i:i+step, None] + lam2[None, :]
            s += (np.exp(-0.5*np.maximum(D2, 0)/V) * V**(-d/2)).sum()
        fh2 = s * norm / N**2
        cross = 0.0
        for (w, mu, S) in zip(self.w, self.mu, self.S):
            if lambdas is None:
                cross += w * _gauss(data - mu, S + H).mean()
            else:
                for l2 in np.unique(lam2):
                    sel = lam2 == l2
                    cross += w * _gauss(data[sel] - mu, S + l2*H).sum() / N
        return fh2 - 2*cross + self.int_f2()

    def psi2(self):
        """Exact matrix int f_ii f_jj (for testing)."""
        d = self.d
        out = np.zeros((d, d))
        for (w1, m1, S1) in zip(self.w, self.mu, self.S):
            for (w2, m2, S2) in zip(self.w, self.mu, self.S):
                # int phi_1'' phi_2'' = 4th derivative of phi_{S1+S2} at m1-m2
                S = S1 + S2
                P = np.linalg.inv(S)
                z = P @ (m1 - m2)
                g = _gauss((m1 - m2)[None], S)[0]
                for i in range(d):
                    for j in range(d):
                        # d^4/dx_i^2 dx_j^2 of N(x;0,S) at x = m1-m2
                        v = (z[i]**2*z[j]**2 - P[i, i]*z[j]**2 - P[j, j]*z[i]**2
                             - 4*P[i, j]*z[i]*z[j] + P[i, i]*P[j, j]
                             + 2*P[i, j]**2)
                        out[i, j] += w1*w2*g*v
        return out


def _gauss(x, S):
    d = S.shape[0]
    P = np.linalg.inv(S)
    q = np.einsum('ni,ij,nj->n', x, P, x)
    return np.exp(-0.5*q) / np.sqrt((2*PI)**d * np.linalg.det(S))


def _diag(*s):
    return np.diag(np.array(s, dtype=float)**2)


def _corr(sd, rho):
    sd = np.asarray(sd, dtype=float)
    C = np.array([[1, rho[0], rho[1]], [rho[0], 1, rho[2]], [rho[1], rho[2], 1]])
    return C * np.outer(sd, sd)


def benchmark_densities():
    """A set of 3d mixtures of increasing difficulty."""
    I = np.eye(3)
    return [
        GaussianMixture('A: standard normal', [1], [[0, 0, 0]], [I]),
        GaussianMixture('B: strongly correlated', [1], [[0, 0, 0]],
                        [_corr([1, 1, 1], [0.9, 0.7, 0.6])]),
        GaussianMixture('C: bimodal', [0.5, 0.5],
                        [[-1.5, 0, 0], [1.5, 0, 0]], [I*0.5**2, I*0.5**2]),
        GaussianMixture('D: three separated modes, different scales', [1/3, 1/3, 1/3],
                        [[0, 0, 0], [18, 0, 0], [15, -18, 5]],
                        [_diag(1, 0.25, 1), _diag(1, 1, 0.5),
                         _diag(1, 0.5, 2)]),
        GaussianMixture('E: skewed', [0.2, 0.2, 0.6],
                        [[0, 0, 0], [0.5, 0.5, 0.5], [1.3, 1.3, 1.3]],
                        [I, I*(2/3)**2, I*(5/9)**2]),
        GaussianMixture('F: kurtotic', [2/3, 1/3],
                        [[0, 0, 0], [0, 0, 0]],
                        [I, I*0.1**2]),
        GaussianMixture('G: bimodal, opposite correlations', [0.5, 0.5],
                        [[-1, -1, 0], [1, 1, 0]],
                        [_corr([0.6, 0.6, 1], [-0.8, 0.3, -0.3]),
                         _corr([0.6, 0.6, 1], [-0.8, -0.3, 0.3])]),
    ]
