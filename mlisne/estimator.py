from abc import ABC, abstractmethod
from pydantic.dataclasses import dataclass # Use pydantic for runtime type-checking
from dataclasses import InitVar
from pathlib import Path
from typing import Tuple, Dict, Union, Sequence, Optional
from sklearn.base import BaseEstimator

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats

from helpers import check_is_fitted, print_summary

class TreatmentIVEstimator(BaseEstimator):
    def __init__(self, seed=0):
        self.seed = seed
        self.__coef = None
        self.__varcov = None
        self.__N = None
        self.__resid = None
        self.__fitted = None
        self.__fit = False
        self.__postest = None
    def fit(self, data: IVEstimatorDataset, qps: np.ndarray, single_nondegen = False) -> None:
        # ==== Compute 2SLS Estimator ====
        # Estimates 2SLS regression of the form:
        #   D = g0(1-I) + g1*Z + g2*QPS(X, delta) + v
        #   Y = b0(1-I) + b1*D + b2*QPS(X, delta) + e
        # Where I is an indicator for whether ML takes only one nondegenerate value in the sample (to avoid multicollinearity).
        # Only observations where QPS %in% (0,1) are used in the estimation.
        # Matrix forms
        #   - W: [[1,...,1], [Z_1, ..., Z_n], [qps_1, ..., qps_n]] (3 x n)
        #   - V: [[1,...,1], [D_1, ..., D_n], [qps_1, ..., qps_n]] (3 x n)
        #   - Y: [Y_1, ..., Y_n] (n x 1)
        #   - e_hat: diag([e_1, ..., e_n]) (n x n) (2nd stage residuals along diagonal)
        # Formulas:
        #   - beta_hat = (W x V')^-1 (W x Y)
        #   - var_cov = (W x V')^-1 (W x e_hat^2 x W')(V x W')^-1

        Y = data.Y
        Z = data.Z
        D = data.D

        # Take indices of the inputs for which QPS_i %in% (0,1)
        obs_tokeep = np.nonzero((qps > 0) & (qps < 1))
        print(f"We will fit on {len(obs_tokeep)} values out of {len(Y)} from the dataset for which the QPS estimation is nondegenerate.")
        assert len(obs_tokeep[0]) > 0

        # Adjusted input matrices for 2SLS
        Y_adj = Y[obs_tokeep]
        D_adj = D[obs_tokeep]
        Z_adj = Z[obs_tokeep]
        QPS_adj = qps[obs_tokeep]
        N_adj = len(Y_adj)

        if single_nondegen: # No constant
            W = np.stack((Z_adj, QPS_adj), axis=0)
            V = np.stack((D_adj, QPS_adj), axis=0)
        else:
            W = np.stack((np.ones(N_adj), Z_adj, QPS_adj), axis=0)
            V = np.stack((np.ones(N_adj), D_adj, QPS_adj), axis=0)

        beta_hat = multi_dot([inv(np.dot(W, V.T)), W, Y_adj]) # Compute 2SLS estimators [b0, b1, b2]

        # Heteroskedasticity robust variance-covariance matrix
        if single_nondegen:
            e_hat = Y_adj - np.multiply(D_adj, beta_hat[0]) - np.multiply(QPS_adj, beta_hat[1])
        else:
            e_hat = Y_adj - beta_hat[0] - np.multiply(D_adj, beta_hat[1]) - np.multiply(QPS_adj, beta_hat[2])
        e_hat_sq = np.diag(e_hat**2)
        var_cov = multi_dot([inv(np.dot(W, V.T)), W, e_hat_sq, W.T, inv(np.dot(V, W.T))])

        self.__coef = beta_hat
        self.__varcov = var_cov
        self.__N = N_adj
        self.__df_resid = self.__N - 2
        self.__resid = e_hat
        self.__fitted = Y_adj - e_hat
        self.__fit = True
        self.__std_error = np.sqrt(np.diag(self.__varcov))
        self.__tstat = np.divide(self.__coef/self.__std_error)
        self.__p = 2 - 2 * stats.t.cdf(abs(self.__tstat), self.__resid)
        self.__upperci = self.__coef + self.__std_error * stats.t.ppf(0.925, self.__df_resid)
        self.__lowerci = self.__coef + self.__std_error * stats.t.ppf(0.025, self.__df_resid)

        # Compute first stage values as well in order to save
        self.__firststage = self._fit_firststage(W.T, D_adj, QPS_adj, single_nondegen)
        self.__postest = None
    @property
    def coef(self):
        # Returns: np array of estimated coefficients, if fitted
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return self.__coef
    @property
    def varcov(self):
        # Returns: variance-covariance matrix, if fitted
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return self.__varcov
    @property
    def fitted(self):
        # Returns: fitted values, if fitted
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return self.__fitted
    @property
    def resid(self):
        # Returns: residuals, if fitted
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return self.__resid
    @property
    def tstat(self):
        # Returns: count of fitted values, if fitted
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return self.__tstat
    @property
    def std_error(self):
        # Returns: count of fitted values, if fitted
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return self.__std_error
    @property
    def p(self):
        # Returns: count of fitted values, if fitted
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return self.__p
    @property
    def n_fit(self):
        # Returns: count of fitted values, if fitted
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return self.__N
    @property
    def ci(self):
        # Returns: Second stage coefficient CI, if fitted
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return np.append(self.__lowerci, self.__upperci, axis=1)
    @property
    def postest(self):
        """Computes post-estimation attributes if not called yet.
        Returns dictionary of attributes.
        """
        if self.__postest is None:
            print("First time calling post-estimation. Computing values...")
            self.__postest = {'resid':self.resid, 'cov':self.varcov, 'cov_type':'robust', 'fitted':self.fitted}
            self.__postest['rss'] = np.sum(self.resid**2)
            self.__postest['s2'] = rss/(self.n_fit - 2) # Unbiased estimate of residual variance
            Y = self.fitted + self.resid
            self.__postest['tss'] = np.sum((Y - np.mean(Y))**2)
            self.__posteest['r2'] = 1 - rss/tss
        return self.__postest
    @property
    def firststage(self):
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...")
            return None
        return np.append(self.__lowerci, self.__upperci, axis=1)
    def _fit_fs_stage(self, X, D, QPS, single_nondegen):
        """Computes post-estimation attributes for first stage regression if not called yet.
        Returns dictionary of first-stage attributes.
        """
        fs_beta_hat = multi_dot([inv(X @ X.T), X.T, D])
        fs_out = {'coef':fs_beta_hat}
        if single_nondegen:
            fs_e_hat = D - np.multiply(X, fs_beta_hat[0]) - np.multiply(QPS, fs_beta_hat[1])
        else:
            fs_e_hat = D - fs_beta_hat[0] - np.multiply(D, fs_beta_hat[1]) - np.multiply(QPS, fs_beta_hat[2])
        fs_rss = np.sum(fs_e_hat**2)
        fs_s2 = rss/(self.n_fit - 2)
        fs_var_cov = s2 * inv(X.T @ X)
        fs_fitted = D - fs_e_hat
        fs_stderror = np.sqrt(np.diag(var_cov))
        fs_tstat = np.divide(fs_beta_hat, fs_stderror)
        fs_p = 2 - 2 * stats.t.cdf(abs(fs_tstat), self.__df_resid)
        fs_upperci = self.fs_beta_hat + self.fs_stderror * stats.t.ppf(0.925, self.__df_resid)
        fs_lowerci = self.fs_beta_hat + self.fs_stderror * stats.t.ppf(0.025, self.__df_resid)
        fs_tss = np.sum((D - np.mean(D))**2)
        fs_r2 = 1 - fs_rss/fs_tss

        fs_out['resid'] = fs_e_hat
        fs_out['rss'] = fs_rss
        fs_out['s2'] = fs_s2
        fs_out['cov'] = fs_var_cov
        fs_out['fitted'] = fs_fitted
        fs_out['std_error'] = fs_stderror
        fs_out['tstat'] = fs_tstat
        fs_out['p'] = fs_p
        fs_out['upperci'] = fs_upperci
        fs_out['loweri'] = fs_lowerci
        fs_out['tss'] = fs_tss
        fs_out['r2'] = fs_r2
        return fs_out
    def __summary(self) -> str:
        """Summary str of model estimation results"""
        ret = "="*50 + "\n"
        ret += "IV TREATMENT ESTIMATION RESULTS\n"
        ret += "="*50 + "\n"
        ret += "\t\tParameter\tStd. Err.\tT-stat\tP-value\tLower CI\tUpper CI\n"
        ret += "-"*50 + "\n"
        coef_list = ['const', 'D', 'QPS']
        for i in len(self.coef):
            coef_str = f"{coef_list[i]}\t\t{self.coef[i]}\t{self.std_error[i]}\t{self.tstat[i]}\t{self.p[i]}\t{self.__lowerci[i]}\t{self.__upperci[i]}\n"
            ret += coef_str
        ret += "="*50 + "\n"
        ret += f"N: {self.n_fit}\n"
        postest = self.postest
        ret += f"Covariance type: {self.postest['cov_type']}\n"
        return ret
    def __repr__(self):
        """Calls post_estimation if post-estimation attributes not computed yet. Prints summary table"""
        if not self.__fit:
            return f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator..."
        return self.__summary()
