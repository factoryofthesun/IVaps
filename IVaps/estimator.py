"""Treatment estimation functions"""
from linearmodels.iv import IV2SLS
from linearmodels.system.model import SUR
from statsmodels.api import add_constant
import numpy as np
import pandas as pd

def estimate_treatment_effect(aps = None, Y = None, Z = None, D = None, data = None, Y_ind = None, Z_ind = None, D_ind = None, aps_ind = None,
                              estimator: str = "2SLS", verbose: bool = True):
    """Main treatment effect estimation function

    Parameters
    -----------
    aps: array-like, default: None
        Array of estimated APS values
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
    aps_ind: int, default: None
        Index of APS variable in `data`
    estimator: str, default: "2SLS"
        Method of IV estimation
    verbose: bool, default: True
        Whether to print output of estimation

    Returns
    -----------
    IVResults
        Fitted IV model object

    Notes
    -----
    Treatment effect is estimated using IV estimation. The default is to use the 2SLS method of estimation, with the equations illustrated below.

    .. math::

        D_i = \\gamma_0(1-I) + \\gamma_1 Z_i + \\gamma_2 p^s(X_i;\\delta) + v_i \\
        Y_i = \\beta_0(1-I) + \\beta_1 D_i + \\beta_2 p^s(X_i;\\delta) + \\epsilon_i

    :math:`\\beta_1` is our causal estimation of the treatment effect. :math:`I` is an indicator for if the ML funtion takes only a single nondegenerate value in the sample.

    aps, Y, Z, D, and data should never have any overlapping columns. This is not checkable through the code, so please double check this when passing in the inputs.

    """
    if data is not None:
        data = np.array(data)

    vals = {"Y": Y, "Z": Z, "D": D, "APS": aps}

    # If `data` given, then use index inputs for values not explicitly passed
    infer = []
    to_del = []
    if data is not None:
        inds = {"Y": Y_ind, "Z": Z_ind, "D": D_ind, "APS": aps_ind}
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
    aps = vals["APS"]

    if aps is None or Y is None or Z is None or D is None:
        raise ValueError("Treatment effect estimation requires all values aps, Y, Z, and D to be passed!")

    lm_inp = pd.DataFrame({"Y": Y, "Z": Z, "D": D, "aps": aps})

    # Use only observations where aps is nondegenerate
    aps = np.array(aps)
    obs_tokeep = np.nonzero((aps > 0) & (aps < 1))
    print(f"We will fit on {len(obs_tokeep[0])} values out of {len(Y)} from the dataset for which the APS estimation is nondegenerate.")
    assert len(obs_tokeep[0]) > 0

    lm_inp = lm_inp.iloc[obs_tokeep[0],:]

    # Check for single non-degeneracy
    single_nondegen = True
    if len(np.unique(aps[obs_tokeep])) > 1:
        single_nondegen = False
        lm_inp = add_constant(lm_inp)

    if estimator == "2SLS":
        if single_nondegen:
            results = IV2SLS(lm_inp['Y'], lm_inp[['aps']], lm_inp['D'], lm_inp['Z']).fit(cov_type='robust')
        else:
            results = IV2SLS(lm_inp['Y'], lm_inp[['const', 'aps']], lm_inp['D'], lm_inp['Z']).fit(cov_type='robust')
    elif estimator == "OLS":
        if single_nondegen:
            results = IV2SLS(lm_inp['Y'], lm_inp[['Z', 'aps']], None, None).fit(cov_type='unadjusted')
        else:
            results = IV2SLS(lm_inp['Y'], lm_inp[['const', 'Z', 'aps']], None, None).fit(cov_type='unadjusted')
    else:
        raise NotImplementedError(f"Estimator option {estimator} not implemented yet!")

    if verbose:
        print(results)

    return results

def estimate_counterfactual_ml(aps = None, Y = None, Z = None, ml_out = None, cf_ml_out = None, data = None, Y_ind = None,
                               Z_ind = None, ml_out_ind = None, cf_ml_out_ind = None, aps_ind = None,
                               cov_type: str = "unadjusted", single_nondegen: bool = False, verbose: bool = True):
    """Estimate counterfactual performance of a new algorithm

    Parameters
    -----------
    aps: array-like, default: None
        Array of estimated APS values
    Y: array-like, default: None
        Array of outcome variables
    Z: array-like, default: None
        Array of treatment recommendations
    ml_out: array-like, default: None
        Original ML function outputs
    cf_ml_out: array-like, default: None
        Counterfactual ML function outputs
    data: array-like, default: None
        2D array of estimation inputs
    Y_ind: int, default: None
        Index of outcome variable in `data`
    Z_ind: int, default: None
        Index of treatment recommendation variable in `data`
    ml_out_ind: int, default: None
        Index of original ML output variable in `data`
    cf_ml_out_ind: int, default: None
        Index of counterfactual ML output variable in `data`
    aps_ind: int, default: None
        Index of APS variable in `data`
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

    Then we take the original ML output ``ML1`` and the counterfactual ML output ``ML2`` and estimate the below value equation.

    .. math::

        \\hat{V}(ML') = \\frac{1}{n} \\sum_{i = 1}^n (Y_i + \\hat{\\beta_{ols}}(ML'(X_i) - ML(X_i))

    """
    if data is not None:
        data = np.array(data)

    vals = {"Y": Y, "Z": Z, "APS": aps , "ml_out": ml_out, "cf_ml_out": cf_ml_out}

    # If `data` given, then use index inputs for values not explicitly passed
    infer = []
    to_del = []
    if data is not None:
        inds = {"Y": Y_ind, "Z": Z_ind, "APS": aps_ind, "ml_out": ml_out_ind, "cf_ml_out": cf_ml_out_ind}
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
    aps = vals["APS"]
    ml_out = np.array(vals["ml_out"])
    cf_ml_out = np.array(vals["cf_ml_out"])

    if aps is None or Y is None or Z is None or ml_out is None or cf_ml_out is None:
        raise ValueError("Treatment effect estimation requires all values aps, Y, Z, D and ML recommendations to be passed!")

    lm_inp = pd.DataFrame({"Y": Y, "Z": Z, "aps": aps})
    if not single_nondegen:
        lm_inp = add_constant(lm_inp)

    ols_results = IV2SLS(lm_inp['Y'], lm_inp[['const', 'Z', 'aps']], None, None).fit(cov_type='unadjusted')

    if verbose:
        print(ols_results)

    b_ols = ols_results.params['Z']
    v = Y + b_ols * (cf_ml_out - ml_out)
    v_score = np.mean(v)

    if verbose:
        print(f"Counterfactual value of new ML function: {v_score}")

    return (v, ols_results)

def covariate_balance_test(aps = None, X = None, Z = None, data = None, X_ind = None, Z_ind = None, aps_ind = None,
                           X_labels = None, cov_type = "robust", verbose: bool = True):
    """Covariate Balance Test

    Parameters
    -----------
    aps: array-like, default: None
        Array of estimated APS values
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
    aps_ind: int, default: None
        Index of APS variable in `data`
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
    This function estimates a system of Seemingly Unrelated Regression (SUR) as defined in the linearmodels package.

    APS, X, Z, and data should never have any overlapping columns. This is not checkable through the code, so please double check this when passing in the inputs.

    For APS, X, Z, either the variables themselves should be passed, or their indices in `data`. If neither is passed then an error is raised.

    """
    # Error checking
    if X is None and (X_ind is None or data is None):
        raise ValueError("covariate_balance_test: No valid data passed for X. You must either pass the variable directly into `X` or its index along with a `data` object.")
    if aps is None and (aps_ind is None or data is None):
        raise ValueError("covariate_balance_test: No valid data passed for aps. You must either pass the variable directly into `X` or its index along with a `data` object.")
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
        if aps_ind is not None:
            aps = data[:, aps_ind]
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
    aps = np.array(aps)
    Z = np.array(Z)

    # Use only observations where aps is nondegenerate
    obs_tokeep = np.nonzero((aps > 0) & (aps < 1))
    print(f"We will run balance testing on {len(obs_tokeep[0])} values out of {len(Z)} from the dataset for which the APS estimation is nondegenerate.")
    assert len(obs_tokeep[0]) > 0
    exog = np.column_stack((Z, aps))
    exog = pd.DataFrame(exog, columns = ['Z', 'aps'])
    exog = exog.iloc[obs_tokeep[0],:]
    X = X[obs_tokeep]
    if cov_type != "robust":
        cov_type = "unadjusted"

    # Check for single non-degeneracy
    single_nondegen = True
    if len(np.unique(aps[obs_tokeep])) > 1:
        single_nondegen = False
        exog = add_constant(exog)

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
        res_dict[x_var]['n'] = mv_ols_res.nobs
        res_dict[x_var]['stderr'] = mv_ols_res.std_errors[f"{x_var}_Z"]
    if mv_test_res is None:
        res_dict['joint'] = {}
        res_dict['joint']['p'] = mv_ols_res.pvalues[f"{X_labels[0]}_Z"]
        res_dict['joint']['t'] = mv_ols_res.tstats[f"{X_labels[0]}_Z"]
    else:
        res_dict['joint'] = {}
        res_dict['joint']['p'] = mv_test_res.results['Z']['stat'].iloc[0, 4]
        res_dict['joint']['f'] = mv_test_res.results['Z']['stat'].iloc[0, 3]

    return (mv_ols_res, res_dict)
