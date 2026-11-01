"""Ön işleme, OC-SIMCA (DD-SIMCA) ve PLS yardımcıları. Uygulamadaki JS bunların birebir karşılığıdır."""
import numpy as np
from scipy.signal import savgol_coeffs
from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression

from load import GRID


class Prep:
    """Bölge kırp (+kenar payı) -> Savitzky-Golay türevi ('valid' konvolüsyon) -> SNV."""

    def __init__(self, lo=850, hi=1500, window=15, poly=2, deriv=1):
        self.lo, self.hi, self.window, self.poly, self.deriv = lo, hi, window, poly, deriv
        half = window // 2
        i0 = np.searchsorted(GRID, lo) - half
        i1 = np.searchsorted(GRID, hi, side="right") + half
        self.sl = slice(i0, i1)
        # np.convolve çekirdeği ters çevirir; savgol_coeffs(use='conv') buna uygun sıralıdır
        self.kernel = savgol_coeffs(window, poly, deriv=deriv, use="conv") if deriv else None
        self.x = GRID[i0 + half:i1 - half]

    def __call__(self, X):
        X = np.atleast_2d(X)[:, self.sl]
        if self.kernel is not None:
            X = np.array([np.convolve(r, self.kernel, mode="valid") for r in X])
        else:
            h = self.window // 2
            X = X[:, h:-h]
        X = (X - X.mean(1, keepdims=True)) / X.std(1, ddof=1, keepdims=True)
        return X

    def export(self):
        return dict(grid_start=float(GRID[0]), grid_step=float(GRID[1] - GRID[0]), grid_n=len(GRID),
                    i0=self.sl.start, i1=self.sl.stop, kernel=None if self.kernel is None else self.kernel.tolist(),
                    half=self.window // 2, lo=self.lo, hi=self.hi, window=self.window, poly=self.poly, deriv=self.deriv)


class DDSimca:
    """Veri güdümlü SIMCA (Pomerantsev & Rodionova 2014): f = Nh*h/h0 + Nq*q/q0 ~ chi2(Nh+Nq)."""

    def __init__(self, ncomp, alpha=0.01):
        self.ncomp, self.alpha = ncomp, alpha

    @staticmethod
    def _dof(v):
        m = v.mean()
        return m, max(1, round(2 * m * m / v.var(ddof=1)))

    def fit(self, X):
        self.mean = X.mean(0)
        Xc = X - self.mean
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.P = Vt[:self.ncomp].T
        self.lam = s[:self.ncomp] ** 2 / (len(X) - 1)
        h, q = self.hq(X)
        self.h0, self.Nh = self._dof(h)
        self.q0, self.Nq = self._dof(q)
        self.fcrit = chi2.ppf(1 - self.alpha, self.Nh + self.Nq)
        return self

    def hq(self, X):
        Xc = X - self.mean
        T = Xc @ self.P
        h = (T ** 2 / self.lam).sum(1)
        q = ((Xc - T @ self.P.T) ** 2).sum(1)
        return h, q

    def f(self, X):
        h, q = self.hq(X)
        return self.Nh * h / self.h0 + self.Nq * q / self.q0

    def predict(self, X):  # True = otantik
        return self.f(X) <= self.fcrit

    def export(self):
        return dict(ncomp=self.ncomp, alpha=self.alpha, mean=self.mean.tolist(), P=self.P.T.tolist(),
                    lam=self.lam.tolist(), h0=self.h0, Nh=self.Nh, q0=self.q0, Nq=self.Nq, fcrit=float(self.fcrit))


class PLS:
    """y_hat = X @ coef + intercept (sklearn ölçekleme katsayılara gömülü)."""

    def __init__(self, ncomp):
        self.ncomp = ncomp

    def fit(self, X, y):
        m = PLSRegression(self.ncomp, scale=False).fit(X, y)
        self.coef = m.coef_.ravel()
        self.intercept = float(np.ravel(m.intercept_)[0])
        return self

    def predict(self, X):
        return X @ self.coef + self.intercept

    def export(self):
        return dict(ncomp=self.ncomp, coef=self.coef.tolist(), intercept=self.intercept)
