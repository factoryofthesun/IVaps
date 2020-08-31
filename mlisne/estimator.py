"""Estimator classes"""
from abc import ABC, abstractmethod
from pydantic.dataclasses import dataclass # Use pydantic for runtime type-checking
from dataclasses import InitVar
from pathlib import Path
from typing import Tuple, Dict, Union, Sequence, Optional
from sklearn.base import BaseEstimator

import warnings
import numpy as np
from numpy.linalg import inv, multi_dot
import pandas as pd
import scipy.stats as stats
from prettytable import PrettyTable

from mlisne.dataset import IVEstimatorDataset
from mlisne.helpers import run_onnx_session

class TreatmentIVEstimator(BaseEstimator):
    """Class to estimate treatment effects using 2SLS method

    The `fit` class method estimates the following equations

    .. math::

        D_i = \gamma_0(1-I) + \gamma_1 Z_i + \gamma_2 p^s(X_i;\delta) + v_i \\
        Y_i = \\beta_0(1-I) + \\beta_1 D_i + \\beta_2 p^s(X_i;\delta) + \epsilon_i

    :math:`\\beta_1` is our causal estimation of the treatment effect. :math:`I` is an indicator for if the ML funtion takes only a single nondegenerate value in the sample.

    """

    def __init__(self):
        self.__coef = None
        self.__varcov = None
        self.__N = None
        self.__resid = None
        self.__fitted = None
        self.__fit = False
        self.__postest = None
    def fit(self, data: IVEstimatorDataset, qps: np.ndarray, single_nondegen: bool = False) -> None:
        """Fit estimator

        Parameters
        -----------
        data: IVEstimatorDataset
            Dataset class loaded with Y, Z, D, and X
        qps: array-like, shape(n_sample,)
            Estimated quasi propensity scores for each observation
        single_nondegen: Boolean
            Indicator for whether the ML model takes on only 1 nondegenerate value in the sample

        """

        Y = data.Y
        Z = data.Z
        D = data.D

        # Take indices of the inputs for which QPS_i %in% (0,1)
        obs_tokeep = np.nonzero((qps > 0) & (qps < 1))
        print(f"We will fit on {len(obs_tokeep[0])} values out of {len(Y)} from the dataset for which the QPS estimation is nondegenerate.")
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
        self.__tstat = self.__coef/self.__std_error
        self.__p = 2 - 2 * stats.t.cdf(abs(self.__tstat), self.__df_resid)
        self.__upperci = self.__coef + self.__std_error * stats.t.ppf(0.975, self.__df_resid)
        self.__lowerci = self.__coef + self.__std_error * stats.t.ppf(0.025, self.__df_resid)
        self.__inputs = {'Y':Y_adj, 'Z':Z_adj, 'D':D_adj, 'QPS':QPS_adj}

        # Compute first stage values as well in order to save time
        self.__firststage = self._fit_firststage(W.T, D_adj, single_nondegen)
        self.__postest = None

    def predict(self, X: np.ndarray):
        """Predict outcome

        Parameters
        -----------
        X: array-like
            Model inputs (D, qps)

        Returns
        -----------
        np.ndarray
            Array of predicted outcomes

        """
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None

        return self.coef[0] + np.sum(self.coef[1:2] * X, axis = 1)

    @property
    def coef(self):
        """np.ndarray: Estimated coefficients"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__coef

    @property
    def varcov(self):
        """np.ndarray: Variance-covariance matrix"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__varcov

    @property
    def fitted(self):
        """np.ndarray: Fitted values"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__fitted

    @property
    def resid(self):
        """np.ndarray: Residuals"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__resid

    @property
    def tstat(self):
        """np.ndarray: T-stat of estimated coefficients"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__tstat

    @property
    def std_error(self):
        """np.ndarray: Standard error of estimated coefficients"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__std_error

    @property
    def p(self):
        """np.ndarray: P-value of estimated coefficients"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__p

    @property
    def n_fit(self):
        """int: Count of fitted values"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__N

    @property
    def ci(self):
        """np.ndarray: Second stage coefficient CI"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return np.stack((self.__lowerci, self.__upperci), axis=1)

    @property
    def inputs(self):
        """np.ndarray: Second stage adjusted inputs"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__inputs

    @property
    def postest(self):
        """Computes post-estimation attributes if not called yet.

        Returns
        -------
        dict
            Dictionary of attributes.

        """
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        elif self.__postest is None:
            print("First time calling post-estimation. Computing values...")
            self.__postest = {'resid':self.resid, 'cov':self.varcov, 'cov_type':'robust', 'fitted':self.fitted}
            rss = np.sum(self.resid**2)
            self.__postest['rss'] = rss
            self.__postest['s2'] = rss/(self.n_fit - 2) # Unbiased estimate of residual variance
            Y = self.fitted + self.resid
            tss = np.sum((Y - np.mean(Y))**2)
            self.__postest['tss'] = tss
            self.__postest['r2'] = 1 - rss/tss
        return self.__postest

    @property
    def firststage(self):
        """dict: Dictionary containing first stage attributes"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__firststage

    def firststage_summary(self):
        """Print summary of model first stage results"""
        fs_res = self.firststage
        x = PrettyTable()
        x.field_names = ["", "Parameter", "Std. Err", "T-Stat", "P-Value", "Lower CI", "Upper CI"]
        coef_list = ['const', 'ML Recommendation', 'QPS']
        for i in range(len(coef_list)):
            row = [coef_list[i], round(fs_res['coef'][i], 4), round(fs_res['std_error'][i], 4), round(fs_res['tstat'][i], 4), round(fs_res['p'][i], 4), round(fs_res['lowerci'][i], 4), round(fs_res['upperci'][i], 4)]
            x.add_row(row)
        print(x)

    def _fit_firststage(self, X, D, single_nondegen):
        """Computes post-estimation attributes for first stage regression if not called yet. Returns dictionary of first-stage attributes."""
        fs_beta_hat = multi_dot([inv(X.T @ X), X.T, D])
        fs_out = {'coef':fs_beta_hat}
        if single_nondegen:
            fs_e_hat = D - np.multiply(X[:,1], fs_beta_hat[0]) - np.multiply(X[:,2], fs_beta_hat[1])
        else:
            fs_e_hat = D - fs_beta_hat[0] - np.multiply(X[:,1], fs_beta_hat[1]) - np.multiply(X[:,2], fs_beta_hat[2])

        fs_e_hat_sq = np.diag(fs_e_hat**2)
        fs_df_resid = self.__df_resid - 1 # There is one more exogenous variable in the first stage OLS
        fs_rss = np.sum(fs_e_hat**2)
        fs_s2 = fs_rss/(self.n_fit - 2)
        fs_var_cov = multi_dot([inv(np.dot(X.T, X)), X.T, fs_e_hat_sq, X, inv(np.dot(X.T, X))]) # Heteroskedastic robust covariance
        fs_fitted = D - fs_e_hat
        fs_stderror = np.sqrt(np.diag(fs_var_cov))
        fs_tstat = np.divide(fs_beta_hat, fs_stderror)
        fs_p = 2 - 2 * stats.t.cdf(abs(fs_tstat), fs_df_resid)
        fs_upperci = fs_beta_hat + fs_stderror * stats.t.ppf(0.925, fs_df_resid)
        fs_lowerci = fs_beta_hat + fs_stderror * stats.t.ppf(0.025, fs_df_resid)
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
        fs_out['lowerci'] = fs_lowerci
        fs_out['ci'] = np.stack((fs_lowerci, fs_upperci), axis=1)
        fs_out['tss'] = fs_tss
        fs_out['r2'] = fs_r2
        return fs_out

    def __summary(self) -> str:
        """Summary str of model estimation results"""
        x = PrettyTable()
        x.field_names = ["", "Parameter", "Std. Err", "T-Stat", "P-Value", "Lower CI", "Upper CI"]
        coef_list = ['const', 'Treatment', 'QPS']
        for i in range(len(self.coef)):
            row = [coef_list[i], round(self.coef[i], 4), round(self.std_error[i], 4), round(self.tstat[i], 4), round(self.p[i], 4), round(self.__lowerci[i], 4), round(self.__upperci[i], 4)]
            x.add_row(row)
        ret = x.get_string()
        return ret

    # def __str__(self) -> str:
    #     ret = "="*70 + "\n"
    #     ret += "IV Treatment Estimation Results\n"
    #     ret += "="*70 + "\n"
    #     ret += "\t\tParameter\tStd. Err.\tT-stat\tP-value\tLower CI\tUpper CI\n"
    #     ret += "-"*70 + "\n"
    #     coef_list = ['const', 'Treatment', 'QPS']
    #     for i in range(len(self.coef)):
    #         coef_str = f"{coef_list[i]}\t\t{round(self.coef[i], 4)}\t{round(self.std_error[i], 4)}\t{round(self.tstat[i], 4)}\t{round(self.p[i], 4)}\t{round(self.__lowerci[i], 4)}\t{round(self.__upperci[i], 4)}\n"
    #         ret += coef_str
    #     ret += "="*70 + "\n"
    #     ret += f"N: {self.n_fit}\n"
    #     postest = self.postest
    #     ret += f"Covariance type: {self.postest['cov_type']}\n"
    #     return ret

    def __repr__(self):
        """Calls post_estimation if post-estimation attributes not computed yet. Prints summary table"""
        if not self.__fit:
            return f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator..."
        return self.__summary()

class CounterfactualMLEstimator(BaseEstimator):
    """Class to estimate counterfactual performance of another algorithm

    The ``fit`` class method estimates the following equation

    .. math::

        Y_i = \\beta_0 + \\beta_1 Z_i + \\beta_2 p^s(X_i;\delta) + \epsilon_i

    :math:`\\beta_1` is our estimated effect of treatment recommendation.

    The ``predict_counterfact`` method takes an ML input (ONNX or user-defined function) and estimates the following value equation

    .. math::

        \hat{V}(ML') = \\frac{1}{n} \sum_{i = 1}^n (Y_i + \hat{\\beta_{ols}}(ML'(X_i) - ML(X_i))

    """

    def __init__(self):
        self.__coef = None
        self.__varcov = None
        self.__N = None
        self.__resid = None
        self.__fitted = None
        self.__fit = False
        self.__postest = None
    def fit(self, data: IVEstimatorDataset, qps: np.ndarray, cov_type: str = "unadjusted") -> None:
        """Fit OLS estimator

        Parameters
        -----------
        data: IVEstimatorDataset
            Dataset class loaded with Y, Z, D, and X
        qps: array-like, shape(n_sample,)
            Estimated quasi propensity scores for each observation
        single_nondegen: Boolean
            Indicator for whether the ML model takes on only 1 nondegenerate value in the sample

        """

        Y = data.Y
        Z = data.Z
        N = Y.shape[0]

        X = np.stack((np.ones(N), Z, qps), axis=0).T

        beta_hat = multi_dot([inv(X.T @ X), X.T, Y])
        e_hat = Y - beta_hat[0] - np.multiply(X[:,1], beta_hat[1]) - np.multiply(X[:,2], beta_hat[2])
        e_hat_sq = np.diag(e_hat**2)
        rss = np.sum(e_hat**2)
        s2 = rss/(N-3)

        if cov_type == "unadjusted":
            var_cov = multi_dot([inv(np.dot(X.T, X)), X.T, e_hat_sq, X, inv(np.dot(X.T, X))])
        elif cov_type == "robust": # Heteroskedastic robust covariance
            var_cov = s2 * inv(np.dot(X.T, X))
        else:
            raise ValueError(f"{cov_type} not a valid covariance type! Please enter either 'unadjusted' or 'robust'.")

        self.__coef = beta_hat
        self.__varcov = var_cov
        self.__N = N
        self.__df_resid = self.__N - 3 # 3 coefficients being estimated
        self.__resid = e_hat
        self.__fitted = Y - e_hat
        self.__fit = True
        self.__std_error = np.sqrt(np.diag(self.__varcov))
        self.__tstat = self.__coef/self.__std_error
        self.__p = 2 - 2 * stats.t.cdf(abs(self.__tstat), self.__df_resid)
        self.__upperci = self.__coef + self.__std_error * stats.t.ppf(0.975, self.__df_resid)
        self.__lowerci = self.__coef + self.__std_error * stats.t.ppf(0.025, self.__df_resid)
        self.__cov_type = cov_type

        self.__postest = None

    def predict(self, X: np.ndarray):
        """Predict outcome

        Parameters
        -----------
        X: array-like
            Model inputs (Z, qps)

        Returns
        -----------
        np.ndarray
            Array of predicted float value scores

        """
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None

        return self.coef[0] + np.sum(self.coef[1:2] * X, axis = 1)

    def predict_counterfact(self, Y: np.ndarray, original_rec: np.ndarray, new_rec: np.ndarray, verbose=True):
        """Predict counterfactual value of a new ML algorithm

        Parameters
        -----------
        Y: array-like
            Outcome data from original treatment
        original_rec: array-like
            Probabilities of treatment recommendation from original ML algorithm
        new_rec: array-like
            Probabilities of treatment recommendation from new ML algorithm
        verbose: boolean, default = True
            Whether to print counterfactual value estimation to console

        Returns
        -----------
        float
            Estimated counterfactual value

        """
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        b_ols = self.coef[1]
        v = np.mean(Y + b_ols * (new_rec - original_rec))
        if verbose:
            print(f"Counterfactual value of new ML function: {v}")
        return v

    @property
    def coef(self):
        """np.ndarray: Estimated coefficients"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__coef

    @property
    def varcov(self):
        """np.ndarray: Variance-covariance matrix"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__varcov

    @property
    def fitted(self):
        """np.ndarray: Fitted values"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__fitted

    @property
    def resid(self):
        """np.ndarray: Residuals"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__resid

    @property
    def tstat(self):
        """np.ndarray: T-stat of estimated coefficients"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__tstat

    @property
    def std_error(self):
        """np.ndarray: Standard error of estimated coefficients"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__std_error

    @property
    def p(self):
        """np.ndarray: P-value of estimated coefficients"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__p

    @property
    def n_fit(self):
        """int: Count of fitted values"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__N

    @property
    def ci(self):
        """np.ndarray: Second stage coefficient CI"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return np.stack((self.__lowerci, self.__upperci), axis=1)

    @property
    def inputs(self):
        """np.ndarray: Second stage adjusted inputs"""
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        return self.__inputs

    @property
    def postest(self):
        """Computes post-estimation attributes if not called yet.

        Returns
        -------
        dict
            Dictionary of attributes

        """
        if not self.__fit:
            warnings.warn(f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator...", stacklevel=2)
            return None
        elif self.__postest is None:
            print("First time calling post-estimation. Computing values...")
            self.__postest = {'resid':self.resid, 'cov':self.varcov, 'cov_type':self.cov_type, 'fitted':self.fitted}
            rss = np.sum(self.resid**2)
            self.__postest['rss'] = rss
            self.__postest['s2'] = rss/(self.n_fit - 3) # Unbiased estimate of residual variance
            Y = self.__fitted + self.resid
            tss = np.sum((Y - np.mean(Y))**2)
            self.__postest['tss'] = tss
            self.__postest['r2'] = 1 - rss/tss
        return self.__postest

    def __summary(self) -> str:
        """Summary str of model estimation results"""
        x = PrettyTable()
        x.field_names = ["", "Parameter", "Std. Err", "T-Stat", "P-Value", "Lower CI", "Upper CI"]
        coef_list = ['const', 'ML Recommendation', 'QPS']
        for i in range(len(self.coef)):
            row = [coef_list[i], round(self.coef[i], 4), round(self.std_error[i], 4), round(self.tstat[i], 4), round(self.p[i], 4), round(self.__lowerci[i], 4), round(self.__upperci[i], 4)]
            x.add_row(row)
        ret = x.get_string()
        return ret

    def __repr__(self):
        """Calls post_estimation if post-estimation attributes not computed yet. Prints summary table"""
        if not self.__fit:
            return f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator..."
        return self.__summary()
