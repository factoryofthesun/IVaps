"""Treatment estimation functions"""
from linearmodels.iv import IV2SLS
from linearmodels.system.model import SUR
from statsmodels.api import add_constant
import numpy as np
import pandas as pd

def estimate_treatment_effect(qps = None, Y = None, Z = None, D = None, data = None, Y_ind = None, Z_ind = None, D_ind = None, qps_ind = None,
                              estimator: str = "2SLS", verbose: bool = True):
    """Main treatment effect estimation function

    Parameters
    -----------
    qps: array-like, default: None
        Array of estimated QPS values
    Y: array-like, default: None
        Array of outcome variables
    Z: array-like, default: None
        Array of treatment recommendations
    D: array-like, default: None
        Array of treatment assignments
    data: array-like, default: None
        2D array of estimation inputs
    Y_ind: int, default: None
        Index of outcome variable in `data`
    Z_ind: int, default: None
        Index of treatment recommendation variable in `data`
    D_ind: int, default: None
        Index of treatment assignment variable in `data`
    qps_ind: int, default: None
        Index of QPS variable in `data`
    estimator: str, default: "2SLS"
        Method of IV estimation
    verbose: bool, default: True
        Whether to print output of estimation

    Returns
    -----------
    tuple(np.ndarray, IVResults)
        Tuple containing array of predicted float value scores and fitted IV results.

    Notes
    -----
    Treatment effect is estimated using IV estimation. The default is to use the 2SLS method of estimation, with the equations illustrated below.

    .. math::

        D_i = \\gamma_0(1-I) + \\gamma_1 Z_i + \\gamma_2 p^s(X_i;\\delta) + v_i \\
        Y_i = \\beta_0(1-I) + \\beta_1 D_i + \\beta_2 p^s(X_i;\\delta) + \\epsilon_i

    :math:`\\beta_1` is our causal estimation of the treatment effect. :math:`I` is an indicator for if the ML funtion takes only a single nondegenerate value in the sample.

    qps, Y, Z, D, and data should never have any overlapping columns. This is not checkable through the code, so please double check this when passing in the inputs.

    """
    if data is not None:
        data = np.array(data)

    vals = {"Y": Y, "Z": Z, "D": D, "QPS": qps}

    # If `data` given, then use index inputs for values not explicitly passed
    infer = []
    to_del = []
    if data is not None:
        inds = {"Y": Y_ind, "Z": Z_ind, "D": D_ind, "QPS": qps_ind}
        for key, val in vals.items():
            if val is None:
                if inds[key] is not None:
                    vals[key] = data[:,inds[key]]
                    to_del.append(inds[key])
                else:
                    infer.append(key)
        data = np.delete(data, to_del, axis=1)
        if len(infer) != 0:
            print(f"Indices for {infer} not explicitly passed. Assuming remaining columns in order {infer}...")
            for i in range(len(infer)):
                vals[infer[i]] = data[:,i]
    Y = vals["Y"]
    Z = vals["Z"]
    D = vals["D"]
    qps = vals["QPS"]

    if qps is None or Y is None or Z is None or D is None:
        raise ValueError("Treatment effect estimation requires all values qps, Y, Z, and D to be passed!")

    lm_inp = pd.DataFrame({"Y": Y, "Z": Z, "D": D, "qps": qps})

    # Use only observations where qps is nondegenerate
    qps = np.array(qps)
    obs_tokeep = np.nonzero((qps > 0) & (qps < 1))
    print(f"We will fit on {len(obs_tokeep[0])} values out of {len(Y)} from the dataset for which the QPS estimation is nondegenerate.")
    assert len(obs_tokeep[0]) > 0

    lm_inp = lm_inp.iloc[obs_tokeep[0],:]

    # Check for single non-degeneracy
    single_nondegen = True
    if len(np.unique(qps[obs_tokeep])) > 1:
        single_nondegen = False
        lm_inp = add_constant(lm_inp)

    if estimator == "2SLS":
        if single_nondegen:
            results = IV2SLS(lm_inp['Y'], lm_inp[['qps']], lm_inp['D'], lm_inp['Z']).fit(cov_type='robust')
        else:
            results = IV2SLS(lm_inp['Y'], lm_inp[['const', 'qps']], lm_inp['D'], lm_inp['Z']).fit(cov_type='robust')
    elif estimator == "OLS":
        if single_nondegen:
            results = IV2SLS(lm_inp['Y'], lm_inp[['Z', 'qps']], None, None).fit(cov_type='unadjusted')
        else:
            results = IV2SLS(lm_inp['Y'], lm_inp[['const', 'Z', 'qps']], None, None).fit(cov_type='unadjusted')
    else:
        raise NotImplementedError(f"Estimator option {estimator} not implemented yet!")

    if verbose:
        print(results)

    return results

def estimate_counterfactual_ml(qps = None, Y = None, Z = None, recs = None, cf_recs = None, data = None, Y_ind = None, Z_ind = None, recs_ind = None, cf_recs_ind = None, qps_ind = None,
                               cov_type: str = "unadjusted", single_nondegen: bool = False, verbose: bool = True):
    """Function to estimate counteractual performance of another algorithm

    Parameters
    -----------
    qps: array-like, default: None
        Array of estimated QPS values
    Y: array-like, default: None
        Array of outcome variables
    Z: array-like, default: None
        Array of treatment recommendations
    recs: array-like, default: None
        Original ML function outputs
    cf_recs: array-like, default: None
        Counterfactual ML function outputs
    data: array-like, default: None
        2D array of estimation inputs
    Y_ind: int, default: None
        Index of outcome variable in `data`
    Z_ind: int, default: None
        Index of treatment recommendation variable in `data`
    recs_ind: int, default: None
        Index of original ML output variable in `data`
    recs_ind: int, default: None
        Index of counterfactual ML output variable in `data`
    qps_ind: int, default: None
        Index of QPS variable in `data`
    estimator: str, default: "2SLS"
        Method of IV estimation
    single_nondegen: bool, default: False
        Indicator for whether the original ML algorithm takes on a single non-degenerate value in the sample
    verbose: bool, default: True
        Whether to print output of estimation

    Returns
    -----------
    tuple(np.ndarray, OLSResults)
        Tuple containing array of predicted float value scores and fitted OLS results.

    Notes
    -----
    The process of estimating counterfactual value works as follows.
    First we fit the below OLS regression using historical recommendations and outcome ``Z`` and ``Y``.

    .. math::

        Y_i = \\beta_0 + \\beta_1 Z_i + \\beta_2 p^s(X_i;\\delta) + \\epsilon_i

    :math:`\\beta_1` is our estimated effect of treatment recommendation.

    The we take the original ML output ``ML1`` and the counterfactual ML output ``ML2`` and estimate the below value equation.

    .. math::

        \\hat{V}(ML') = \\frac{1}{n} \\sum_{i = 1}^n (Y_i + \\hat{\\beta_{ols}}(ML'(X_i) - ML(X_i))

    """
    if data is not None:
        data = np.array(data)

    vals = {"Y": Y, "Z": Z, "QPS": qps , "recs": recs, "cf_recs": cf_recs}

    # If `data` given, then use index inputs for values not explicitly passed
    infer = []
    to_del = []
    if data is not None:
        inds = {"Y": Y_ind, "Z": Z_ind, "QPS": qps_ind, "recs": recs_ind, "cf_recs": cf_recs_ind}
        for key, val in vals.items():
            if val is None:
                if inds[key] is not None:
                    vals[key] = data[:,inds[key]]
                    to_del.append(inds[key])
                else:
                    infer.append(key)
        data = np.delete(data, to_del, axis=1)
        if len(infer) != 0:
            print(f"Indices for {infer} not explicitly passed. Assuming remaining columns in order {infer}...")
            for i in range(len(infer)):
                vals[infer[i]] = data[:,i]

    Y = vals["Y"]
    Z = vals["Z"]
    qps = vals["QPS"]
    recs = np.array(vals["recs"])
    cf_recs = np.array(vals["cf_recs"])

    if qps is None or Y is None or Z is None or recs is None or cf_recs is None:
        raise ValueError("Treatment effect estimation requires all values qps, Y, Z, D and ML recommendations to be passed!")

    lm_inp = pd.DataFrame({"Y": Y, "Z": Z, "qps": qps})
    if not single_nondegen:
        lm_inp = add_constant(lm_inp)

    ols_results = IV2SLS(lm_inp['Y'], lm_inp[['const', 'Z', 'qps']], None, None).fit(cov_type='unadjusted')

    if verbose:
        print(ols_results)

    b_ols = ols_results.params['Z']
    v = Y + b_ols * (cf_recs - recs)
    v_score = np.mean(v)

    if verbose:
        print(f"Counterfactual value of new ML function: {v_score}")

    return (v, ols_results)

def covariate_balance_test(qps = None, X = None, Z = None, data = None, X_ind = None, Z_ind = None, qps_ind = None,
                           X_labels = None, cov_type = "robust", verbose: bool = True):
    """Covariate Balance Test

    Parameters
    -----------
    qps: array-like, default: None
        Array of estimated QPS values
    X: array-like, default: None
        Array of covariates to test
    Z: array-like, default: None
        Array of treatment recommendations
    data: array-like, default: None
        2D array of estimation inputs
    X_ind: int/array_of_int, default: None
        Indices/indices of covariates in `data`
    Z_ind: int, default: None
        Index of treatment recommendation variable in `data`
    qps_ind: int, default: None
        Index of QPS variable in `data`
    X_labels: array-like, default: None
        Array of string labels to associate with each covariate
    cov_type: str, default: "robust"
        Covariance type of SUR. Any value other than "robust" defaults to simple (nonrobust) covariance.
    verbose: bool, default: True
        Whether to print output for each test

    Returns
    -----------
    tuple(SystemResults, dict(X_label, dict(stat_label, value)))
        Tuple containing the fitted SUR model results and a dictionary containing the results of covariate balance estimation for each covariate as well as the joint hypothesis.

    Notes
    -----
    This function esimates a system of Seemingly Unrelated Regression (SUR) as defined in the linearmodels package.

    QPS, X, Z, and data should never have any overlapping columns. This is not checkable through the code, so please double check this when passing in the inputs.

    For QPS, X, Z, either the variables themselves should be passed, or their indices in `data`. If neither is passed then an error is raised.

    """
    # Error checking
    if X is None and (X_ind is None or data is None):
        raise ValueError("covariate_balance_test: No valid data passed for X. You must either pass the variable directly into `X` or its index along with a `data` object.")
    if qps is None and (qps_ind is None or data is None):
        raise ValueError("covariate_balance_test: No valid data passed for qps. You must either pass the variable directly into `X` or its index along with a `data` object.")
    if Z is None and (Z is None or data is None):
        raise ValueError("covariate_balance_test: No valid data passed for Z. You must either pass the variable directly into `X` or its index along with a `data` object.")
    if X_labels is not None:
        if X.ndim == 1:
            if len(X_labels) > 1:
                raise ValueError(f"Column labels {X_labels} not the same length as inputs.")
        else:
            if len(X_labels) != X.shape[1]:
                raise ValueError(f"Column labels {X_labels} not the same length as inputs.")

    # Construct covariate balance inputs
    if data is not None:
        data = np.array(data)
        if X_ind is not None:
            X = data[:, X_ind]
        if qps_ind is not None:
            qps = data[:, qps_ind]
        if Z_ind is not None:
            Z = data[:, Z_ind]
    if isinstance(X, np.ndarray):
        if X.ndim == 1 and X_labels is None:
            X_labels = ['X1']
        elif X_labels is None:
            X_labels = [f"X{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns = X_labels)
    elif X_labels is not None:
        X.columns = X_labels
    else:
        X_labels = X.columns
    qps = np.array(qps)
    Z = np.array(Z)
    exog = np.column_stack((Z, qps))
    exog = pd.DataFrame(exog, columns = ['Z', 'qps'])
    exog = add_constant(exog)
    if cov_type != "robust":
        cov_type = "unadjusted"

    # Covariate balance test
    mv_ols_res = SUR.multivariate_ls(X, exog).fit(cov_type = cov_type)
    if verbose == True:
        print(mv_ols_res)

    # Joint hypothesis test: use multivariate_OLS from statsmodels
    # Edge case: single variable then joint test is the same as the original
    if len(X_labels) > 1:
        from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
        mv_ols_joint = _MultivariateOLS(X, exog).fit()
        L = np.zeros((1,3))
        L[:,1] = 1
        mv_test_res = mv_ols_joint.mv_test([("Z", L)])
    else:
        mv_test_res = None

    # Compile results
    res_dict = {}
    for x_var in X_labels:
        res_dict[x_var] = {}
        res_dict[x_var]['coef'] = mv_ols_res.params[f"{x_var}_Z"]
        res_dict[x_var]['p'] = mv_ols_res.pvalues[f"{x_var}_Z"]
        res_dict[x_var]['t'] = mv_ols_res.tstats[f"{x_var}_Z"]
    if mv_test_res is None:
        res_dict['joint'] = {}
        res_dict['joint']['p'] = mv_ols_res.pvalues[f"{X_labels[0]}_Z"]
        res_dict['joint']['t'] = mv_ols_res.tstats[f"{X_labels[0]}_Z"]
    else:
        res_dict['joint'] = {}
        res_dict['joint']['p'] = mv_test_res.results['Z']['stat'].iloc[0, 4]
        res_dict['joint']['f'] = mv_test_res.results['Z']['stat'].iloc[0, 3]

    return (mv_ols_res, res_dict)
