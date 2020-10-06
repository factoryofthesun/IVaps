from pathlib import Path
from typing import Tuple, Dict, Set, Union, Sequence, Optional
import onnxruntime as rt
import warnings
import numpy as np
import pandas as pd

from mlisne import run_onnx_session

def _computeQPSOld(X_ci: np.ndarray, X_di: np.ndarray, L_ind: np.ndarray, L_vals: np.ndarray,
                types: Tuple[np.dtype, np.dtype], S: int, delta: float, mu: np.ndarray, sigma: np.ndarray, sess: rt.InferenceSession,
                input_type: int, input_names: Tuple[str, str], fcn, **kwargs):
    """Compute QPS for a single row of data

    Quasi-propensity score estimation involves taking draws :math:`X_c^1, \\ldots,X_c^S` from the uniform distribution on :math:`N(X_{ci}, \\delta)`, where :math:`N(X_{ci},\\delta)` is the :math:`p_c` dimensional ball centered at :math:`X_{ci}` with radius :math:`\\delta`.

    :math:`X_c^1, \\ldots,X_c^S` are destandardized before passed for ML inference. The estimation equation is :math:`p^s(X_i;\\delta) = \\frac{1}{S} \\sum_{s=1}^{S} ML(X_c^s, X_{di})`.

    Parameters
    -----------
    X_ci: array-like, shape(n_continuous,)
        1D vector of standardized continuous inputs
    X_di: array-like, shape(n_discrete,), default: None
        1D vector of discrete inputs
    L_ind: array-like, shape(n_mixed,), default: None
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
    # ===================================================================================================================
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
    return qps

def _estimate_qps_onnx(onnx: str, X_c = None, X_d = None, data = None, C: Sequence = None, D: Sequence = None, L: Dict[int, Set] = None,
                      S: int = 100, delta: float = 0.8, seed: int = None, types: Tuple[np.dtype, np.dtype] = (None, None), input_type: int = 1,
                      input_names: Tuple[str, str]=("c_inputs", "d_inputs"), fcn = None, vectorized: bool = False, cpu: bool = False,
                      parallel: bool = False, nprocesses: int = None, nchunks: int = None, **kwargs):
    """Estimate QPS for given dataset and ONNX model

    Parameters
    -----------
    onnx: str
        String path to ONNX model
    X_c: array-like, default: None
        1D/2D vector of continuous input variables
    X_d: array-like, default: None
        1D/2D vector of discrete input variables
    data: array-like, default: None
        Dataset containing ML input variables
    C: array-like, default: None
        Integer column indices for continous variables
    D: array-like, default: None
        Integer column indices for discrete variables
    L: Dict[int, Set]
        Dictionary with keys as indices of X_c and values as sets of discrete values
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
    cpu: bool, default False
        Run inference on CPU; defaults to GPU if available
    parallel: bool, default: False
        Whether to parallelize the QPS estimation
    nprocesses: int, default: None
        Number of processes to parallelize. Defaults to number of processors on machine.
    nchunks: int, default: None
        Number of chunks to send to each worker. Defaults to 14*nprocesses.

    Returns
    -----------
    np.ndarray
        Array of estimated QPS for each observation in sample

    """
    # Set X_c and X_d based on inputs
    if X_c is None and data is None:
        raise ValueError("QPS estimation requires continuous data!")

    # Prioritize explicitly passed variables
    if X_c is not None:
        X_c = np.array(X_c)
    if X_d is not None:
        X_d = np.array(X_d)

    if data is not None:
        data = np.array(data)

    # If X_c not given, but data is, then we assume all of data is X_c
    if X_c is None and X_d is not None and data is not None:
        print("`X_c` not given but both `X_d` and `data` given. We will assume that all the variables in `data` are continuous.")
        X_c = data

    # If X_d not given, but data is, then we assume all of data is X_d
    if X_c is not None and X_d is None and data is not None:
        print("`X_d` not given but both `X_c` and `data` given. We will assume that all the variables in `data` are discrete.")
        X_d = data

    # If both X_c and X_d are none, then use indices
    if X_c is None and X_d is None:
        if C is None and D is None:
            print("`data` given but no indices passed. We will assume that all the variables in `data` are continuous.")
            X_c = data
        elif C is None:
            if isinstance(D, int):
                d_len = 1
            else:
                d_len = len(D)
            X_d = data[:,D]
            if d_len >= data.shape[1]:
                raise ValueError(f"Passed discrete indices of length {d_len} for input data of shape {data.shape}. Continuous variables are necessary to conduct QPS estimation.")
            else:
                print(f"Passed discrete indices of length {d_len} for input data of shape {data.shape}. Remaining columns of `data` will be assumed to be continuous variables.")
                X_c = np.delete(data, D, axis = 1)
        elif D is None:
            if isinstance(C, int):
                c_len = 1
            else:
                c_len = len(C)
            X_c = data[:,C]
            if c_len < data.shape[1]:
                print(f"Passed continuous indices of length {c_len} for input data of shape {data.shape}. Remaining columns of `data` will be assumed to be discrete variables.")
                X_d = np.delete(data, C, axis = 1)
        else:
            X_c = data[:,C]
            X_d = data[:,D]

    # Force X_c to be 2d array
    if X_c.ndim == 1:
        X_c = X_c[:,np.newaxis]

    # Vectorize decision function if not
    if fcn is not None and vectorized == False:
        fcn = np.vectorize(fcn)

    # === Preprocess mixed variables ===
    if L is not None:
        L_keys = np.array(list(L.keys()))
        L_vals = np.array(list(L.values()))

        # Get indices of mixed vars to replace for each row
        mixed_og_rows = [np.where(np.isin(X_c[:,L_keys[i]], list(L_vals[i])))[0] for i in range(len(L_keys))] # List of row indices for each mixed variable column
        mixed_og_cols = [np.repeat(L_keys[i], len(mixed_og_rows[i])) for i in range(len(mixed_og_rows))]
        mixed_og_inds = (np.concatenate(mixed_og_rows), np.concatenate(mixed_og_cols))

        # Save original discrete values
        mixed_og_vals = X_c[mixed_og_inds]

        # Replace values at indices with NA
        X_c[mixed_og_inds] = np.nan

    # If types not given, then infer from data
    types = list(types)
    if types[0] is None:
        types[0] = X_c.dtype
    if types[1] is None:
        if X_d is not None:
            types[1] = X_d.dtype

    # === Standardize continuous variables ===
    # Formula: (X_ik - u_k)/o_k; k represents a continuous variable
    mu = np.nanmean(X_c, axis=0)
    sigma = np.nanstd(X_c, axis=0)
    X_c = np.apply_along_axis(lambda row: np.divide(np.subtract(row, mu), sigma), axis=1, arr=X_c)

    if seed is not None:
        np.random.seed(seed)
    sess = rt.InferenceSession(onnx)

    # Set CPU provider if specified
    if cpu == True:
        print("Available providers:", sess.get_providers())
        print("Setting to default CPU provider...")
        sess.set_providers(["CPUExecutionProvider"])

    QPS_vec = []
    for i in range(X_c.shape[0]):
        X_di = None
        X_ci = None
        L_ind_i = None
        L_val_i = None
        if X_d is not None:
            X_di = X_d[i,]
        if L is not None:
            row_inds = np.where(mixed_og_inds[0] == i)
            L_ind_i = mixed_og_inds[1][row_inds]
            L_val_i = mixed_og_vals[row_inds]

        QPS_vec.append(_computeQPSOld(X_c[i,], X_di, L_ind_i, L_val_i, types, S, delta, mu, sigma, sess, input_type, input_names, fcn, **kwargs)) # Compute QPS for each individual i

    QPS_vec = np.array(QPS_vec)
    return QPS_vec

def _computeUserQPSOld(X_ci: np.ndarray, X_di: np.ndarray, L_ind: np.ndarray, L_vals: np.ndarray,
                    ml, S: int, delta: float, mu: np.ndarray, sigma: np.ndarray,
                    pandas:  bool, pandas_cols: Sequence, order: Sequence, reorder: Sequence, **kwargs):
    """Compute QPS for a single row of data, using a user-defined input function.
    Quasi-propensity score estimation involves taking draws :math:`X_c^1, \\ldots,X_c^S` from the uniform distribution on :math:`N(X_{ci}, \\delta)`, where :math:`N(X_{ci},\\delta)` is the :math:`p_c` dimensional ball centered at :math:`X_{ci}` with radius :math:`\\delta`.
    :math:`X_c^1, \\ldots,X_c^S` are destandardized before passed for ML inference. The estimation equation is :math:`p^s(X_i;\\delta) = \\frac{1}{S} \\sum_{s=1}^{S} ML(X_c^s, X_{di})`.
    Parameters
    -----------
    X_ci: array-like, shape(n_continuous,)
        1D vector of standardized continuous inputs
    X_di: array-like, shape(n_discrete,), default: None
        1D vector of discrete inputs
    L_ind: array-like, shape(n_mixed,), default: None
        1D vector of original mixed discrete indices
    L_vals: array-like, shape(n_mixed,), default: None
        1D vector of original mixed discrete values
    ml: Object
        User-defined vectorized ML function
    S: int
        Number of draws
    delta: float
        Radius of sampling ball
    mu: array-like, shape(n_continuous,)
        1D vector of means of continuous variables
    sigma: array-like, shape(n_continuous,)
        1D vector of standard deviations of continuous variables
    pandas: bool, default: False
        Whether to convert input to pandas DataFrame before sending into function
    pandas_cols: Sequence, default: None
        Column names for pandas input. Pandas defaults to integer names.
    order: Sequence
        Reording the columns after ordering into [cts vars, discrete vars]
    reorder: Sequence, default: False
        Indices to reorder the data assuming original order `order`
    **kwargs: keyword arguments to pass into user function
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

    # Run ML inference ----------------------------------------------------------------------------------------------------------
    # We will assume that ML always takes a single concatenated matrix as input
    if X_di is None:
        inputs = destandard_draws
    else:
        X_d_long = np.tile(X_di, (destandard_draws.shape[0], 1))
        inputs = np.append(destandard_draws, X_d_long, axis=1)

    # Reorder if specified
    if order is not None:
        inputs = inputs[:,order]
    if reorder is not None:
        inputs = inputs[:,reorder]

    # Create pandas input if specified
    if pandas:
        inputs = pd.DataFrame(inputs, columns = pandas_cols)

    ml_out = ml(inputs, **kwargs)
    qps = np.mean(ml_out)
    return qps

def _get_og_order(n, C, D):
    order = None
    if C is None and D is None:
        pass
    elif C is None:
        order = []
        c_len = n - len(D)
        c_ind = 0
        for i in range(n):
            if i in D:
                order.append(c_ind + c_len)
                c_ind += 1
            else:
                order.append(i - c_ind)
    else:
        order = []
        c_len = len(C)
        c_ind = 0
        for i in range(n):
            if i in C:
                order.append(i - c_ind)
            else:
                order.append(c_ind + c_len)
                c_ind += 1
    return order


def _estimate_qps_user_defined(ml, X_c = None, X_d = None, data = None, C: Sequence = None, D: Sequence = None, L: Dict[int, Set] = None,
                              S: int = 100, delta: float = 0.8, seed: int = None, pandas: bool = False, pandas_cols: Sequence = None,
                              keep_order: bool = False, reorder: Sequence = None, parallel: bool = False, nprocesses: int = None, nchunks: int = None, **kwargs):
    """Estimate QPS for given dataset and user defined ML function
    Parameters
    -----------
    ml: Object
        User defined ml function
    X_c: array-like, default: None
        1D/2D vector of continuous input variables
    X_d: array-like, default: None
        1D/2D vector of discrete input variables
    data: array-like, default: None
        Dataset containing ML input variables
    C: array-like, default: None
        Integer column indices for continous variables
    D: array-like, default: None
        Integer column indices for discrete variables
    L: Dict[int, Set]
        Dictionary with keys as indices of X_c and values as sets of discrete values
    S: int, default: 100
        Number of draws for each QPS estimation
    delta: float, default: 0.8
        Radius of sampling ball
    seed: int, default: None
        Seed for sampling
    pandas: bool, default: False
        Whether to cast inputs into pandas dataframe
    pandas_cols: Sequence, default: None
        Columns names for dataframe input
    keep_order: bool, default: False
        Whether to maintain the column order if data passed as a single 2D array
    reorder: Sequence, default: False
        Indices to reorder the data assuming original order [X_c, X_d]
    parallel: bool, default: False
        Whether to parallelize the QPS estimation
    nprocesses: int, default: None
        Number of processes to parallelize. Defaults to number of processors on machine.
    nchunks: int, default: None
        Number of chunks to send to each worker. Defaults to 14*nprocesses.
    **kwargs: keyword arguments to pass into user function
    Returns
    -----------
    np.ndarray
        Array of estimated QPS for each observation in sample
    Notes
    ------
    The arguments `keep_order`, `reorder`, and `pandas_cols` are applied sequentially, in that order. This means that if `keep_order` is set, then `reorder` will reorder the columns from the original column order as `data`. `pandas_cols` will then be the names of the new ordered dataset.
    The default ordering of inputs is [X_c, X_d], where the continuous variables and discrete variables will be in the original order regardless of how their input is passed. If `reorder` is called without `keep_order`, then the reordering will be performed on this default ordering.
    Parallelization uses the `ProcessPoolExecutor` module from concurrent.futures, which will NOT be able to deal with execution on GPU. If the user function enables inference on GPU, then it is recommended to implement parallelization within the user function as well.
    The optimal settings for nprocesses and nchunks are specific to each machine, and it is highly recommended that the user pass these arguments to maximize the performance boost. `This SO thread <https://stackoverflow.com/questions/42074501/python-concurrent-futures-processpoolexecutor-performance-of-submit-vs-map>`_ recommends setting nchunks to be 14 * # of workers for optimal performance.
    """

    # Set X_c and X_d based on inputs
    if X_c is None and data is None:
        raise ValueError("QPS estimation requires continuous data!")

    # Prioritize explicitly passed variables
    if X_c is not None:
        X_c = np.array(X_c)
    if X_d is not None:
        X_d = np.array(X_d)

    if data is not None:
        data = np.array(data)

    # If X_c not given, but data is, then we assume all of data is X_c
    if X_c is None and X_d is not None and data is not None:
        print("`X_c` not given but both `X_d` and `data` given. We will assume that all the variables in `data` are continuous.")
        X_c = data

    # If X_d not given, but data is, then we assume all of data is X_d
    if X_c is not None and X_d is None and data is not None:
        print("`X_d` not given but both `X_c` and `data` given. We will assume that all the variables in `data` are discrete.")
        X_d = data

    # If both X_c and X_d are none, then use indices
    order = None
    if X_c is None and X_d is None:
        # Save original order if keep order in place
        if keep_order:
            order = _get_og_order(data.shape[1], C, D)
        if C is None and D is None:
            print("`data` given but no indices passed. We will assume that all the variables in `data` are continuous.")
            X_c = data
        elif C is None:
            if isinstance(D, int):
                d_len = 1
            else:
                d_len = len(D)
            X_d = data[:,D]
            if d_len >= data.shape[1]:
                raise ValueError(f"Passed discrete indices of length {d_len} for input data of shape {data.shape}. Continuous variables are necessary to conduct QPS estimation.")
            else:
                print(f"Passed discrete indices of length {d_len} for input data of shape {data.shape}. Remaining columns of `data` will be assumed to be continuous variables.")
                X_c = np.delete(data, D, axis = 1)
        elif D is None:
            if isinstance(C, int):
                c_len = 1
            else:
                c_len = len(C)
            X_c = data[:,C]
            if c_len < data.shape[1]:
                print(f"Passed continuous indices of length {c_len} for input data of shape {data.shape}. Remaining columns of `data` will be assumed to be discrete variables.")
                X_d = np.delete(data, C, axis = 1)
        else:
            X_c = data[:,C]
            X_d = data[:,D]

    # Force X_c to be 2d array
    if X_c.ndim == 1:
        X_c = X_c[:,np.newaxis]

    # === Preprocess mixed variables ===
    if L is not None:
        L_keys = np.array(list(L.keys()))
        L_vals = np.array(list(L.values()))

        # Get indices of mixed vars to replace for each row
        mixed_og_rows = [np.where(np.isin(X_c[:,L_keys[i]], list(L_vals[i])))[0] for i in range(len(L_keys))] # List of row indices for each mixed variable column
        mixed_og_cols = [np.repeat(L_keys[i], len(mixed_og_rows[i])) for i in range(len(mixed_og_rows))]
        mixed_og_inds = (np.concatenate(mixed_og_rows), np.concatenate(mixed_og_cols))

        # Save original discrete values
        mixed_og_vals = X_c[mixed_og_inds]

        # Replace values at indices with NA
        X_c[mixed_og_inds] = np.nan

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
    # Parallelize if set
    if parallel == True:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        import multiprocessing as mp
        from traceback import print_exc
        from itertools import repeat
        from functools import partial

        if X_d is None:
            iter_d = repeat(None)
        else:
            iter_d = iter(X_d)
        if L is None:
            iter_L_ind = repeat(None)
            iter_L_val = repeat(None)
        else:
            iter_L_ind = iter(mixed_og_inds)
            iter_L_val = iter(mixed_og_vals)

        computeUserQPS_frozen = partial(_computeUserQPSOld, ml = ml, S = S, delta = delta, mu = mu, sigma = sigma, pandas = pandas, pandas_cols = pandas_cols,
                                                        order = order, reorder = reorder, **kwargs)
        pool = mp.Pool(nprocesses)

        if nprocesses is None:
            workers = "default (# processors)"
            nprocesses = mp.cpu_count()
        else:
            workers = nprocesses
        if nchunks is None:
            nchunks = nprocesses * 14
        print(f"Running QPS estimation with {workers} workers...")

        chunksize = X_c.shape[0]//nchunks
        # Chunksize should be >= 1
        if chunksize < 1:
            chunksize = 1
        iter_args = zip(iter(X_c), iter_d, iter_L_ind, iter_L_val)
        QPS_vec = pool.starmap(computeUserQPS_frozen, iter_args, chunksize = chunksize)
        pool.close()
    else:
        for i in range(X_c.shape[0]):
            X_di = None
            X_ci = None
            L_ind_i = None
            L_val_i = None
            if X_d is not None:
                X_di = X_d[i,]
            if L is not None:
                row_inds = np.where(mixed_og_inds[0] == i)
                L_ind_i = mixed_og_inds[1][row_inds]
                L_val_i = mixed_og_vals[row_inds]
            QPS_vec.append(_computeUserQPSOld(X_c[i,], X_di, L_ind_i, L_val_i,
                            ml, S, delta, mu, sigma, pandas, pandas_cols, order, reorder, **kwargs)) # Compute QPS for each individual i

    QPS_vec = np.array(QPS_vec)
    return QPS_vec
