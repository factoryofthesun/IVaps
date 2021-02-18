"""Treatment estimation functions"""
from linearmodels.iv import IV2SLS
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

        D_i = \gamma_0(1-I) + \gamma_1 Z_i + \gamma_2 p^s(X_i;\delta) + v_i \\
        Y_i = \\beta_0(1-I) + \\beta_1 D_i + \\beta_2 p^s(X_i;\delta) + \epsilon_i

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

        Y_i = \\beta_0 + \\beta_1 Z_i + \\beta_2 p^s(X_i;\delta) + \epsilon_i

    :math:`\\beta_1` is our estimated effect of treatment recommendation.

    The we take the original ML output ``ML1`` and the counterfactual ML output ``ML2`` and estimate the below value equation.

    .. math::

        \hat{V}(ML') = \\frac{1}{n} \sum_{i = 1}^n (Y_i + \hat{\\beta_{ols}}(ML'(X_i) - ML(X_i))

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
