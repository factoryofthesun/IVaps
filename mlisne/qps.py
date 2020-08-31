"""QPS estimation functions"""
from pathlib import Path
from typing import Tuple, Dict, Set, Union, Sequence, Optional
import onnxruntime as rt
import warnings
import numpy as np
import pandas as pd

from mlisne.dataset import IVEstimatorDataset
from mlisne.helpers import run_onnx_session

def _computeQPS(X_ci: np.ndarray, types: Sequence[np.dtype], S: int, delta: float, mu: np.ndarray, sigma: np.ndarray,
                sess: rt.InferenceSession, input_type: int, input_names: Tuple[str, str], X_di: np.ndarray = None,
                L_ind: np.ndarray = None, L_vals: np.ndarray = None, fcn = None, **kwargs):
    """Compute QPS for a single row of data

    Quasi-propensity score estimation involves taking draws :math:`X_c^1, \\ldots,X_c^S` from the uniform distribution on :math:`N(X_{ci}, \\delta)`, where :math:`N(X_{ci},\\delta)` is the :math:`p_c` dimensional ball centered at :math:`X_{ci}` with radius :math:`\\delta`.

    :math:`X_c^1, \\ldots,X_c^S` are destandardized before passed for ML inference. The estimation equation is :math:`p^s(X_i;\\delta) = \\frac{1}{S} \\sum_{s=1}^{S} ML(X_c^s, X_{di})`.

    Parameters
    -----------
    X_ci: array-like, shape(n_continuous,)
        1D vector of standardized continuous inputs
    X_di: array-like, shape(n_discrete,), default: None
        1D vector of discrete inputs
    L_i: array-like, shape(n_mixed,), default: None
        1D vector of original mixed discrete indices
    L_vals: array-like, shape(n_mixed,), default: None
        1D vector of original mixed discrete values
    types: list-like, length(2)
        Numpy dtypes for continuous and discrete data
    S: int
        Number of draws
    delta: float
        Radius of sampling ball
    mu: array-like, shape(n_continuous,)
        1D vector of means of continuous variables
    sigma: array-like, shape(n_continuous,)
        1D vector of standard deviations of continuous variables
    sess: onnxruntime InferenceSession
        Session for running inference on loaded ONNX model
    input_type: 1 or 2
        Whether the model takes continuous/discrete inputs together or separately
    input_names: tuple, length(2)
        Names of input nodes if separate continuous and discrete inputs
    fcn: Object, default = None
        Vectorized decision function to wrap ML output
    **kwargs: keyword arguments to pass into decision function

    Returns
    -----------
    float
        Estimated qps for the observation row

    """
    p_c = len(X_ci) # Number of continuous variables
    na_inds = np.where(np.isnan(X_ci))[0] # Indices of NA values

    # If all continuous vars are NA then we can't compute QPS
    if p_c == len(na_inds):
        return np.nan

    delta_vec = np.array([delta] * p_c)
    standard_draws = np.random.normal(size=(S,p_c)) # S draws from standard normal
    standard_draws[:,na_inds] = np.nan # NEW FEATURE: missing values among continuous variables -- exclude from uniform sampling
    scaled_draws = np.apply_along_axis(lambda row: np.divide(row, np.sqrt(np.nansum(row**2))), axis=1, arr=standard_draws) # Scale each draw by its distance from center
    u = np.random.uniform(low=[0] * S, high = [1] * S)**(1/(p_c - len(na_inds))) # S draws from Unif(0,1), then sent to 1/p_c power
    uniform_draws = scaled_draws * u[:, None] * delta_vec + X_ci # Scale by sampled u and ball mean/radius to get the final uniform draws (S x p_c)

    # De-standardize each of the variables
    destandard_draws = np.add(np.multiply(uniform_draws, sigma), mu) # This applies the transformations element-wise to each row of uniform_draws

    # Add back the original discrete mixed values
    if L_ind is not None:
        destandard_draws[:,L_ind] = L_vals

    # Run ONNX inference ----------------------------------------------------------------------------------------------------------
    cts_type = types[0]
    disc_type = types[1]

    # Multi-output models are typically in the order [label, probabilities], so this is what we'll assume for now
    if len(sess.get_outputs()) > 1:
        label_name = sess.get_outputs()[1].name
    else:
        label_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name

    # Adapt input based on settings
    if X_di is None:
        inputs = destandard_draws.astype(cts_type)
        ml_out = run_onnx_session([inputs], sess, [input_name], [label_name], fcn, **kwargs)[0]
    else:
        X_d_long = np.tile(X_di, (destandard_draws.shape[0], 1))
        if input_type == 2:
            disc_inputs = X_d_long.astype(disc_type)
            cts_inputs = destandard_draws.astype(cts_type)
            ml_out = run_onnx_session([cts_inputs, disc_inputs], sess, input_names, [label_name], fcn, **kwargs)
        else:
            # If input type = 1, then coerce all to the continuous type
            inputs = np.append(destandard_draws, X_d_long, axis=1).astype(cts_type)
            ml_out = run_onnx_session([inputs], sess, [input_name], [label_name], fcn, **kwargs)

    qps = np.mean(ml_out)
    return(qps)

def estimate_qps(X: IVEstimatorDataset, ML_onnx: str, S: int = 100, delta: float = 0.8, seed: int = None,
                 types: Tuple[np.dtype, np.dtype] = (None, None), input_type: int = 1,
                 input_names: Tuple[str, str]=("c_inputs", "d_inputs"), fcn = None, vectorized = False, **kwargs):
    """Estimate QPS for given dataset and ONNX model

    Parameters
    -----------
    X: IVEstimatorDataset
        Dataset with loaded historical treatment data
    ML_onnx: str
        String path to ONNX model
    S: int, default: 100
        Number of draws for each QPS estimation
    delta: float, default: 0.8
        Radius of sampling ball
    seed: int, default: None
        Seed for sampling
    types: Tuple[np.dtype, np.dtype], default: (None, None)
        Numpy dtypes for continuous and discrete data; by default types are inferred
    input_type: int, default: 1
        Whether the model takes continuous/discrete inputs together (1) or separately (2)
    input_names: Tuple[str,str], default: ("c_inputs", "d_inputs")
        Names of input nodes of ONNX model
    fcn: Object, default: None
        Decision function to apply to ML output
    vectorized: bool, default: False
        Indicator for whether decision function is already vectorized

    Returns
    -----------
    np.ndarray
        Array of estimated QPS for each observation in sample

    """

    X_c = X.X_c
    X_d = X.X_d
    L = X.L

    # Vectorize decision function if not
    if fcn is not None and vectorized == False:
        fcn = np.vectorize(fcn)

    # === Preprocess mixed variables ===
    if L is not None:
        L_keys = np.array(list(L.keys()))
        L_vals = np.array(list(L.values()))

        # Get indices of mixed vars to replace for each row
        mixed_og_inds = [L_keys[np.where(X_c[i,L_keys] == L_vals)[0]] for i in range(X_c.shape[0])]
        # Save original discrete values
        mixed_og_vals = [X_c[i,mixed_og_inds[i]] for i in range(len(mixed_og_inds))]
        # Replace values at indices with NA
        for i in range(len(mixed_og_inds)):
            X_c[i,mixed_og_inds[i]] = np.nan

    # If types not given, then infer from data
    types = list(types)
    if types[0] is None:
        types[0] = X.X_c.dtype
    if types[1] is None:
        if X_d is not None:
            types[1] = X.X_d.dtype

    # === Standardize continuous variables ===
    # Formula: (X_ik - u_k)/o_k; k represents a continuous variable
    mu = np.nanmean(X_c, axis=0)
    sigma = np.nanstd(X_c, axis=0)
    X_c = np.apply_along_axis(lambda row: np.divide(np.subtract(row, mu), sigma), axis=1, arr=X_c)

    # TODO: parallelize the QPS estimation across individuals (seems like the most natural way to do it)
    #   - How does this work with GPU? Can we somehow partition the parallel jobs on the same GPU.
    if seed is not None:
        np.random.seed(seed)
    sess = rt.InferenceSession(ML_onnx)
    QPS_vec = []
    for i in range(X_c.shape[0]):
        X_di = None
        X_ci = None
        L_ind_i = None
        L_val_i = None
        if X_d is not None:
            X_di = X_d[i,]
        if L is not None:
            L_ind_i = mixed_og_inds[i]
            L_val_i = mixed_og_vals[i]

        QPS_vec.append(_computeQPS(X_c[i,], types, S, delta, mu, sigma, sess, X_di = X_di, L_ind = L_ind_i,
                        L_vals = L_val_i, input_type=input_type, input_names=input_names, fcn = fcn, **kwargs)) # Compute QPS for each individual i
    QPS_vec = np.array(QPS_vec)
    return QPS_vec

def _computeUserQPS(X_ci: np.ndarray, ml, S: int, delta: float, mu: np.ndarray, sigma: np.ndarray,
                    X_di: np.ndarray = None, **kwargs):
    """Compute QPS for a single row of data, using a user-defined input function.

    Quasi-propensity score estimation involves taking draws :math:`X_c^1, \\ldots,X_c^S` from the uniform distribution on :math:`N(X_{ci}, \\delta)`, where :math:`N(X_{ci},\\delta)` is the :math:`p_c` dimensional ball centered at :math:`X_{ci}` with radius :math:`\\delta`.

    :math:`X_c^1, \\ldots,X_c^S` are destandardized before passed for ML inference. The estimation equation is :math:`p^s(X_i;\\delta) = \\frac{1}{S} \\sum_{s=1}^{S} ML(X_c^s, X_{di})`.

    Parameters
    -----------
    X_ci: array-like, shape(n_continuous,)
        1D vector of standardized continuous inputs
    X_di: array-like, shape(n_discrete,), default: None
        1D vector of discrete inputs
    S: int
        Number of draws
    delta: float
        Radius of sampling ball
    mu: array-like, shape(n_continuous,)
        1D vector of means of continuous variables
    sigma: array-like, shape(n_continuous,)
        1D vector of standard deviations of continuous variables
    **kwargs: keyword arguments to pass into user function

    Returns
    -----------
    float
        Estimated qps for the observation row

    """
    p_c = len(X_ci) # Number of continuous variables
    delta_vec = np.array([delta] * p_c)
    standard_draws = np.random.normal(size=(S,p_c)) # S draws from standard normal
    scaled_draws = np.apply_along_axis(lambda row: np.divide(row, np.sqrt(np.sum(row**2))), axis=1, arr=standard_draws) # Scale each draw by its distance from center
    u = np.random.uniform(low=[0] * S, high = [1] * S)**(1/p_c) # S draws from Unif(0,1), then sent to 1/p_c power
    uniform_draws = scaled_draws * u[:, None] * delta_vec + X_ci # Scale by sampled u and ball mean/radius to get the final uniform draws (S x p_c)

    # De-standardize each of the variables
    destandard_draws = np.add(np.multiply(uniform_draws, sigma), mu) # This applies the transformations element-wise to each row of uniform_draws

    # Run ML inference ----------------------------------------------------------------------------------------------------------
    # We will assume that ML always takes a single concatenated matrix as input
    if X_di is None:
        inputs = destandard_draws
    else:
        X_d_long = np.tile(X_di, (destandard_draws.shape[0], 1))
        inputs = np.append(destandard_draws, X_d_long, axis=1)

    ml_out = ml(inputs, **kwargs)
    qps = np.mean(ml_out)
    return(qps)

def estimate_qps_user_defined(X: IVEstimatorDataset, ml, S: int = 100, delta: float = 0.8, seed: int = None, **kwargs):
    """Estimate QPS for given dataset and user defined ML function

    Parameters
    -----------
    X: IVEstimatorDataset
        Dataset with loaded historical treatment data
    ml: Object
        User defined ml function
    S: int, default: 100
        Number of draws for each QPS estimation
    delta: float, default: 0.8
        Radius of sampling ball
    seed: int, default: None
        Seed for sampling
    **kwargs: keyword arguments to pass into user function

    Returns
    -----------
    np.ndarray
        Array of estimated QPS for each observation in sample

    """

    X_c = X.X_c
    X_d = X.X_d

    # === Standardize continuous variables ===
    # Formula: (X_ik - u_k)/o_k; k represents a continuous variable
    mu = np.mean(X_c, axis=0)
    sigma = np.std(X_c, axis=0)
    X_c = np.apply_along_axis(lambda row: np.divide(np.subtract(row, mu), sigma), axis=1, arr=X_c)

    # TODO: parallelize the QPS estimation across individuals (seems like the most natural way to do it)
    #   - How does this work with GPU? Can we somehow partition the parallel jobs on the same GPU.
    if seed is not None:
        np.random.seed(seed)

    QPS_vec = []
    for i in range(X_c.shape[0]):
        QPS_vec.append(_computeUserQPS(X_c[i,], ml, S, delta, mu, sigma, X_di = X_d[i,], **kwargs)) # Compute QPS for each individual i
    QPS_vec = np.array(QPS_vec)
    return QPS_vec
