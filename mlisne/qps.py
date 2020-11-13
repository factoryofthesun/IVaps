"""QPS estimation functions"""
from pathlib import Path
from typing import Tuple, Dict, Set, Union, Sequence, Optional
import onnxruntime as rt
import warnings
import numpy as np
import pandas as pd
import os
import gc

from mlisne import run_onnx_session

def _computeQPS(onnx, X_c: np.ndarray, X_d: np.ndarray, L_inds: Tuple[np.ndarray, np.ndarray], L_vals: np.ndarray,
                types: Tuple[np.dtype, np.dtype], S: int, delta: float, mu: np.ndarray, sigma: np.ndarray,
                input_type: int, input_names: Tuple[str, str], fcn, cpu: bool, parallel: bool, **kwargs):
    """Compute QPS for a single row of data

    Quasi-propensity score estimation involves taking draws :math:`X_c^1, \\ldots,X_c^S` from the uniform distribution on :math:`N(X_{ci}, \\delta)`, where :math:`N(X_{ci},\\delta)` is the :math:`p_c` dimensional ball centered at :math:`X_{ci}` with radius :math:`\\delta`.

    :math:`X_c^1, \\ldots,X_c^S` are destandardized before passed for ML inference. The estimation equation is :math:`p^s(X_i;\\delta) = \\frac{1}{S} \\sum_{s=1}^{S} ML(X_c^s, X_{di})`.

    Parameters
    -----------
    onnx: str
        Path to saved ONNX model
    X_c: array-like
        1D vector of standardized continuous inputs
    X_d: array-like
        1D vector of discrete inputs
    L_inds: tuple
        Tuple of indices for mixed values in X_c
    L_vals: array-like
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
    input_type: 1 or 2
        Whether the model takes continuous/discrete inputs together or separately
    input_names: tuple, length(2)
        Names of input nodes if separate continuous and discrete inputs
    fcn: Object
        Vectorized decision function to wrap ML output
    cpu: bool
        Whether to run inference on CPU
    parallel: bool
        Whether function is being called in a parallelized process
    **kwargs: keyword arguments to pass into decision function

    Returns
    -----------
    float
        Estimated qps for the observation row

    """
    # =================================== QPS estimation with full matrix form ==================================
    p_c = X_c.shape[1]
    nrows = X_c.shape[0]
    na_inds = np.where(np.isnan(X_c))

    # For each row in X_c, run separate sampling procedure
    standard_draws = np.random.normal(size = (nrows, S, p_c))
    standard_draws[na_inds[0], :, na_inds[1]] = np.nan
    scaled_draws = np.apply_along_axis(lambda row: np.divide(row, np.sqrt(np.nansum(row**2))), axis=-1, arr=standard_draws)
    na_counts = np.apply_along_axis(lambda row: np.sum(row), axis = 1, arr = np.isnan(X_c)) # Get NA counts per row
    non_na_cts = p_c - na_counts # Count of non-na draws for each row
    u = np.random.uniform(size=(nrows, S))**(np.divide(1, non_na_cts, out=np.array([np.nan]*len(na_counts)), where=non_na_cts!=0))[:,np.newaxis] # nrows x S draws from Unif(0,1) then send to power depending on NA in row
    uniform_draws = scaled_draws * u[:,:,None] * delta + X_c[:,None,:] # Scale by sampled u and ball mean/radius to get the final uniform draws (S x p_c)

    # De-standardize each of the variables
    destandard_draws = np.add(np.multiply(uniform_draws, sigma), mu) # This applies the transformations continuous variable-wise

    # Add back the original discrete mixed values
    if L_inds is not None:
        destandard_draws[L_inds[0], :, L_inds[1]] = L_vals[:,np.newaxis]

    # Collapse to 2D for inference
    inference_draws = destandard_draws.reshape((nrows*S, p_c))

    # Run ONNX inference ----------------------------------------------------------------------------------------------------------
    sess = rt.InferenceSession(onnx)
    options = rt.SessionOptions()

    # Set CPU provider
    if cpu == True:
        sess.set_providers(["CPUExecutionProvider"])
    else: # If on GPU then put input and output on CUDA: don't need to implement this yet
        if sess.get_providers()[0] == "CUDAExecutionProvider":
            pass

    # Set session threads if parallelizing
    if parallel == True:
        os.environ["OMP_NUM_THREADS"] = '1'
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1

    cts_type = types[0]
    disc_type = types[1]

    # Multi-output models are typically in the order [label, probabilities], so this is what we'll assume for now
    if len(sess.get_outputs()) > 1:
        label_name = sess.get_outputs()[1].name
    else:
        label_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name

    # Adapt input based on settings
    if X_d is None:
        inputs = inference_draws.astype(cts_type)
        ml_out = run_onnx_session([inputs], sess, [input_name], [label_name], fcn, **kwargs)
    else:
        X_d_long = np.repeat(X_d, destandard_draws.shape[1], axis=0)
        if input_type == 2:
            disc_inputs = X_d_long.astype(disc_type)
            cts_inputs = inference_draws.astype(cts_type)
            ml_out = run_onnx_session([cts_inputs, disc_inputs], sess, input_names, [label_name], fcn, **kwargs)
        else:
            # If input type = 1, then coerce all to the continuous type
            inputs = np.append(inference_draws, X_d_long, axis=1).astype(cts_type)
            ml_out = run_onnx_session([inputs], sess, [input_name], [label_name], fcn, **kwargs)

    # Explicitly delete ONNX session
    del sess

    # Return means of every S rows
    qps = np.cumsum(ml_out, 0)[S-1::S]/float(S)
    qps[1:] = qps[1:] - qps[:-1]
    return qps

def estimate_qps_onnx(onnx: str, X_c = None, X_d = None, data = None, C: Sequence = None, D: Sequence = None, L: Dict[int, Set] = None,
                      S: int = 100, delta: float = 0.8, seed: int = None, types: Tuple[np.dtype, np.dtype] = (None, None), input_type: int = 1,
                      input_names: Tuple[str, str]=("c_inputs", "d_inputs"), fcn = None, vectorized: bool = False, cpu: bool = False, iobound: bool = False,
                      parallel: bool = False, nprocesses: int = None, ntasks: int = 1, **kwargs):
    """Estimate QPS for given dataset and ONNX model

    Quasi-propensity score estimation involves taking draws :math:`X_c^1, \\ldots,X_c^S` from the uniform distribution on :math:`N(X_{ci}, \\delta)`, where :math:`N(X_{ci},\\delta)` is the :math:`p_c` dimensional ball centered at :math:`X_{ci}` with radius :math:`\\delta`.

    :math:`X_c^1, \\ldots,X_c^S` are destandardized before passed for ML inference. The estimation equation is :math:`p^s(X_i;\\delta) = \\frac{1}{S} \\sum_{s=1}^{S} ML(X_c^s, X_{di})`.

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
    ntasks: int, default: 1
        Number of tasks to send to each worker process.

    Returns
    -----------
    np.ndarray
        Array of estimated QPS for each observation in sample

    Notes
    ------
    X_c, X_d, and data should never have any overlapping columns. This is not checkable through the code, so please double check this when passing in the inputs.

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

    # Force data to be 2d arrays
    if X_c.ndim == 1:
        X_c = X_c[:,np.newaxis]
    if X_d is not None:
        if X_d.ndim == 1:
            X_d = X_d[:,np.newaxis]

    # Vectorize decision function if not
    if fcn is not None and vectorized == False:
        fcn = np.vectorize(fcn)

    # === Preprocess mixed variables ===
    mixed_og_inds = None
    mixed_og_vals = None
    if L is not None:
        L_keys = np.array(list(L.keys()))
        L_vals = np.array(list(L.values()))

        # Get indices of mixed vars to replace for each row
        mixed_og_rows = [np.where(np.isin(X_c[:,L_keys[i]], list(L_vals[i])))[0] for i in range(len(L_keys))] # List of row indices for each mixed variable column
        mixed_og_cols = [np.repeat(L_keys[i], len(mixed_og_rows[i])) for i in range(len(mixed_og_rows))]
        mixed_rows = np.concatenate(mixed_og_rows)
        mixed_cols = np.concatenate(mixed_og_cols)
        mixed_og_inds = (mixed_rows, mixed_cols)

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

    # If parallelizing, then force inference on CPU
    if parallel == True:
        cpu = True

        # # Need to force Windows implementation of spawning on Linux
        # import multiprocess.context as ctx
        # ctx._force_start_method('spawn')

        import pathos
        from functools import partial
        from itertools import repeat

        computeQPS_frozen = partial(_computeQPS, types = types, S = S, delta = delta, mu = mu, sigma = sigma, input_type = input_type,
                                    input_names = input_names, fcn = fcn, cpu = cpu, parallel = parallel, **kwargs)
        mp = pathos.helpers.mp
        p = mp.Pool(nprocesses)
        #p = pathos.pools._ProcessPool(nprocesses)

        if nprocesses is None:
            workers = "default (# processors)"
            nprocesses = mp.cpu_count()
        else:
            workers = nprocesses
        print(f"Running QPS estimation with {workers} workers...")

        # Split input arrays into chunked rows
        nchunks = ntasks * nprocesses
        X_c_split = np.array_split(X_c, nchunks)
        iter_c = iter(X_c_split)
        if X_d is None:
            iter_d = repeat(None)
        else:
            iter_d = iter(np.array_split(X_d, nchunks))
        if L is None:
            iter_L_ind = repeat(None)
            iter_L_val = repeat(None)
        else:
            # Split indices depending on which chunk they fall into
            chunksizes = np.append([0], np.cumsum([c.shape[0] for c in X_c_split]))
            chunked_inds = [(mixed_rows[np.where(np.isin(mixed_rows, range(chunksizes[i], chunksizes[i+1])))] - chunksizes[i],
                             mixed_cols[np.where(np.isin(mixed_rows, range(chunksizes[i], chunksizes[i+1])))]) for i in range(len(chunksizes) - 1)]
            chunked_vals = [mixed_og_vals[np.where(np.isin(mixed_rows, range(chunksizes[i], chunksizes[i+1])))] for i in range(len(chunksizes) - 1)]
            iter_L_ind = iter(chunked_inds)
            iter_L_val = iter(chunked_vals)

        iter_args = zip(repeat(onnx), iter_c, iter_d, iter_L_ind, iter_L_val)
        p_out = p.starmap(computeQPS_frozen, iter_args)
        p.close()
        p.join()
        qps_vec = np.concatenate(p_out)
    else:
        qps_vec = _computeQPS(onnx, X_c, X_d, mixed_og_inds, mixed_og_vals, types, S, delta, mu, sigma, input_type, input_names, fcn, cpu, parallel, **kwargs) # Compute QPS for each individual i
        qps_vec = np.array(qps_vec)
    gc.collect()
    return qps_vec

def _computeUserQPS(X_c: np.ndarray, X_d: np.ndarray, L_inds: np.ndarray, L_vals: np.ndarray, ml, S: int, delta: float, mu: np.ndarray, sigma: np.ndarray,
                    pandas:  bool, pandas_cols: Sequence, order: Sequence, reorder: Sequence, **kwargs):
    """Compute QPS for a single row of data, using a user-defined input function.

    Quasi-propensity score estimation involves taking draws :math:`X_c^1, \\ldots,X_c^S` from the uniform distribution on :math:`N(X_{ci}, \\delta)`, where :math:`N(X_{ci},\\delta)` is the :math:`p_c` dimensional ball centered at :math:`X_{ci}` with radius :math:`\\delta`.

    :math:`X_c^1, \\ldots,X_c^S` are destandardized before passed for ML inference. The estimation equation is :math:`p^s(X_i;\\delta) = \\frac{1}{S} \\sum_{s=1}^{S} ML(X_c^s, X_{di})`.

    Parameters
    -----------
    X_c: array-like
        1D vector of standardized continuous inputs
    X_d: array-like
        1D vector of discrete inputs
    L_inds: tuple
        Tuple of indices for mixed values in X_c
    L_vals: array-like
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
    pandas: bool
        Whether to convert input to pandas DataFrame before sending into function
    pandas_cols: Sequence
        Column names for pandas input. Pandas defaults to integer names.
    order: Sequence
        Reording the columns after ordering into [cts vars, discrete vars]
    reorder: Sequence
        Indices to reorder the data assuming original order `order`
    seed: int
        Numpy random seed
    **kwargs: keyword arguments to pass into user function

    Returns
    -----------
    float
        Estimated qps for the observation row

    """
    # =================================== QPS estimation with full matrix form ===================================
    p_c = X_c.shape[1]
    nrows = X_c.shape[0]
    na_inds = np.where(np.isnan(X_c))

    # For each row in X_c, run separate sampling procedure
    standard_draws = np.random.normal(size = (nrows, S, p_c))
    standard_draws[na_inds[0], :, na_inds[1]] = np.nan
    scaled_draws = np.apply_along_axis(lambda row: np.divide(row, np.sqrt(np.nansum(row**2))), axis=-1, arr=standard_draws)
    na_counts = np.apply_along_axis(lambda row: np.sum(row), axis = 1, arr = np.isnan(X_c)) # Get NA counts per row
    non_na_cts = p_c - na_counts # Count of non-na draws for each row
    u = np.random.uniform(size=(nrows, S))**(np.divide(1, non_na_cts, out=np.array([np.nan]*len(na_counts)), where=non_na_cts!=0))[:,np.newaxis] # nrows x S draws from Unif(0,1) then send to power depending on NA in row
    uniform_draws = scaled_draws * u[:,:,None] * delta + X_c[:,None,:] # Scale by sampled u and ball mean/radius to get the final uniform draws (S x p_c)

    # De-standardize each of the variables
    destandard_draws = np.add(np.multiply(uniform_draws, sigma), mu) # This applies the transformations continuous variable-wise

    # Add back the original discrete mixed values
    if L_inds is not None:
        destandard_draws[L_inds[0], :, L_inds[1]] = L_vals[:,np.newaxis]

    # Collapse to 2D for inference
    inference_draws = destandard_draws.reshape((nrows*S, p_c))

    # Run ML inference ----------------------------------------------------------------------------------------------------------
    # We will assume that ML always takes a single concatenated matrix as input
    if X_d is None:
        inputs = inference_draws
    else:
        X_d_long = np.repeat(X_d, destandard_draws.shape[1], axis=0)
        inputs = np.append(inference_draws, X_d_long, axis=1)

    # Reorder if specified
    if order is not None:
        inputs = inputs[:,order]
    if reorder is not None:
        inputs = inputs[:,reorder]

    # Create pandas input if specified
    if pandas:
        inputs = pd.DataFrame(inputs, columns = pandas_cols)
    ml_out = np.squeeze(np.array(ml(inputs, **kwargs)))

    # Return means of every S rows
    qps = np.cumsum(ml_out, 0)[S-1::S]/float(S)
    qps[1:] = qps[1:] - qps[:-1]
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


def estimate_qps_user_defined(ml, X_c = None, X_d = None, data = None, C: Sequence = None, D: Sequence = None, L: Dict[int, Set] = None,
                              S: int = 100, delta: float = 0.8, seed: int = None, pandas: bool = False, pandas_cols: Sequence = None,
                              keep_order: bool = False, reorder: Sequence = None, parallel: bool = False, nprocesses: int = None, ntasks: int = 1, **kwargs):
    """Estimate QPS for given dataset and user defined ML function

    Quasi-propensity score estimation involves taking draws :math:`X_c^1, \\ldots,X_c^S` from the uniform distribution on :math:`N(X_{ci}, \\delta)`, where :math:`N(X_{ci},\\delta)` is the :math:`p_c` dimensional ball centered at :math:`X_{ci}` with radius :math:`\\delta`.

    :math:`X_c^1, \\ldots,X_c^S` are destandardized before passed for ML inference. The estimation equation is :math:`p^s(X_i;\\delta) = \\frac{1}{S} \\sum_{s=1}^{S} ML(X_c^s, X_{di})`.

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
    ntasks: int, default: 1
        Number of tasks to send to each worker process.
    **kwargs: keyword arguments to pass into user function

    Returns
    -----------
    np.ndarray
        Array of estimated QPS for each observation in sample

    Notes
    ------
    X_c, X_d, and data should never have any overlapping columns. This is not checkable through the code, so please double check this when passing in the inputs.

    The arguments `keep_order`, `reorder`, and `pandas_cols` are applied sequentially, in that order. This means that if `keep_order` is set, then `reorder` will reorder the columns from the original column order as `data`. `pandas_cols` will then be the names of the new ordered dataset.

    The default ordering of inputs is [X_c, X_d], where the continuous variables and discrete variables will be in the original order regardless of how their input is passed. If `reorder` is called without `keep_order`, then the reordering will be performed on this default ordering.

    Parallelization uses the `Pool` module from pathos, which will NOT be able to deal with execution on GPU. If the user function enables inference on GPU, then it is recommended to implement parallelization within the user function as well.

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
    if X_d is not None:
        if X_d.ndim == 1:
            X_d = X_d[:,np.newaxis]

    # === Preprocess mixed variables ===
    mixed_og_inds = None
    mixed_og_vals = None
    if L is not None:
        L_keys = np.array(list(L.keys()))
        L_vals = np.array(list(L.values()))

        # Get indices of mixed vars to replace for each row
        mixed_og_rows = [np.where(np.isin(X_c[:,L_keys[i]], list(L_vals[i])))[0] for i in range(len(L_keys))] # List of row indices for each mixed variable column
        mixed_og_cols = [np.repeat(L_keys[i], len(mixed_og_rows[i])) for i in range(len(mixed_og_rows))]
        mixed_rows = np.concatenate(mixed_og_rows)
        mixed_cols = np.concatenate(mixed_og_cols)
        mixed_og_inds = (mixed_rows, mixed_cols)

        # Save original discrete values
        mixed_og_vals = X_c[mixed_og_inds]

        # Replace values at indices with NA
        X_c[mixed_og_inds] = np.nan

    # === Standardize continuous variables ===
    # Formula: (X_ik - u_k)/o_k; k represents a continuous variable
    mu = np.nanmean(X_c, axis=0)
    sigma = np.nanstd(X_c, axis=0)
    X_c = np.apply_along_axis(lambda row: np.divide(np.subtract(row, mu), sigma), axis=1, arr=X_c)

    if seed is not None:
        np.random.seed(seed)

    # If parallelizing, then force inference on CPU
    if parallel == True:
        cpu = True
        import pathos
        from functools import partial
        from itertools import repeat

        computeUserQPS_frozen = partial(_computeUserQPS, ml = ml, S = S, delta = delta, mu = mu, sigma = sigma, pandas = pandas,
                                    pandas_cols = pandas_cols, order = order, reorder = reorder, **kwargs)
        mp = pathos.helpers.mp
        p = mp.Pool(nprocesses)

        if nprocesses is None:
            workers = "default (# processors)"
            nprocesses = mp.cpu_count()
        else:
            workers = nprocesses
        print(f"Running QPS estimation with {workers} workers...")

        # Split input arrays into chunked rows
        nchunks = ntasks * nprocesses
        X_c_split = np.array_split(X_c, nchunks)
        iter_c = iter(X_c_split)
        if X_d is None:
            iter_d = repeat(None)
        else:
            iter_d = iter(np.array_split(X_d, nchunks))
        if L is None:
            iter_L_ind = repeat(None)
            iter_L_val = repeat(None)
        else:
            # Split indices depending on which chunk they fall into
            chunksizes = np.append([0], np.cumsum([c.shape[0] for c in X_c_split]))
            chunked_inds = [(mixed_rows[np.where(np.isin(mixed_rows, range(chunksizes[i], chunksizes[i+1])))] - chunksizes[i],
                             mixed_cols[np.where(np.isin(mixed_rows, range(chunksizes[i], chunksizes[i+1])))]) for i in range(len(chunksizes) - 1)]
            chunked_vals = [mixed_og_vals[np.where(np.isin(mixed_rows, range(chunksizes[i], chunksizes[i+1])))] for i in range(len(chunksizes) - 1)]
            iter_L_ind = iter(chunked_inds)
            iter_L_val = iter(chunked_vals)

        iter_args = zip(iter_c, iter_d, iter_L_ind, iter_L_val)
        p_out = p.starmap(computeUserQPS_frozen, iter_args)
        p.close()
        p.join()
        qps_vec = np.concatenate(p_out)

    else:
        qps_vec = _computeUserQPS(X_c, X_d, mixed_og_inds, mixed_og_vals, ml, S, delta, mu, sigma, pandas, pandas_cols, order, reorder, **kwargs) # Compute QPS for each individual i
        qps_vec = np.array(qps_vec)
    return qps_vec
