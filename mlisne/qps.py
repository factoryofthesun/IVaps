from pathlib import Path
from typing import Tuple, Dict, Union, Sequence, Optional
import onnxruntime as rt
import warnings
import numpy as np
import pandas as pd

from mlisne.dataset import IVEstimatorDataset

# Description:
#   - Sample from uniform distribution on delta-ball centered on input vector using Method 2 Appendix B4
#   - De-standardize and compute QPS = avg(ML(X_c^s, X_di)), s = 1 to S
# Inputs:
#   - X_ci: continuous vars for individual i (p_c x 1)
#   - X_di: discrete vars for individual i (p_d x 1)
#   - S: # draws
#   - delta: radius of ball
#   - mu: mean of continuous variables (p_c x 1)
#   - sigma: std of continuous variables (p_c x 1)
#   - sess: ONNX Runtime session that returns a numpy array of outcomes
#   - types: tuple of data types for ONNX input (cts_type, disc_type)
#   - input_type: 1 for single np.ndarray ONNX input, 2 for inputs split by continuous and discrete inputs

# Output: p^s(X_i; delta) (scalar within [0,1])
def _computeQPS(X_ci: np.ndarray, types: Sequence[np.dtype], S: int, delta: int, mu: float, sigma: float, sess: rt.InferenceSession,
                input_type: int, input_names: Tuple[str, str], X_di: np.ndarray = None):
    p_c = len(X_ci) # Number of continuous variables
    delta_vec = np.array([delta] * p_c)
    standard_draws = np.random.normal(size=(S,p_c)) # S draws from standard normal
    scaled_draws = np.apply_along_axis(lambda row: np.divide(row, np.sqrt(np.sum(row**2))), axis=1, arr=standard_draws) # Scale each draw by its distance from center
    u = np.random.uniform(low=[0] * S, high = [1] * S)**(1/p_c) # S draws from Unif(0,1), then sent to 1/p_c power
    uniform_draws = scaled_draws * u[:, None] * delta_vec + X_ci # Scale by sampled u and ball mean/radius to get the final uniform draws (S x p_c)

    # De-standardize each of the variables
    destandard_draws = np.add(np.multiply(uniform_draws, sigma), mu) # This applies the transformations element-wise to each row of uniform_draws

    # Run ONNX inference ----------------------------------------------------------------------------------------------------------
    cts_type = types[0]
    disc_type = types[1]
    label_name = "output_probability" # For now we will assume the output probability label is always "output_probability"
    input_name = sess.get_inputs()[0].name

    # Adapt input based on settings
    if X_di is None:
        inputs = destandard_draws.astype(cts_type)
        ml_out = sess.run([label_name], {input_name: inputs})[0]
    else:
        X_d_long = np.tile(X_di, (destandard_draws.shape[0], 1))
        if input_type == 2:
            disc_inputs = X_d_long.astype(disc_type)
            cts_inputs = destandard_draws.astype(cts_type)
            ml_out = sess.run([label_name], {input_names[0]: cts_inputs, input_names[1]: disc_inputs})[0]
        else:
            # If input type = 1, then coerce all to the continuous type
            inputs = np.append(destandard_draws, X_d_long, axis=1).astype(cts_type)
            ml_out = sess.run([label_name], {input_name: inputs})[0]

    # Account for case in which output probabilities are in dictionary of class labels
    if isinstance(ml_out[0], Dict):
        qps = np.mean(np.array([d[1] for d in ml_out]))
    else:
        qps = np.mean(ml_out)
    return(qps)

# Description:
#   - Standardize continuous variables
#   - Estimate QPS for each individual (row) in input data
# Inputs:
#   - X_c: continuous variables matrix (n x p_c)
#   - X_di: discrete variables matrix (n x p_c)
#   - S: # draws
#   - delta: radius of ball
#   - ML_onnx: path to ML ONNX object that can take a 2d np array of inputs
#   - seed: random seed
#   - input_type: "single" for single ONNX input, "double" for ONNX input that separates discrete and continuous inputs

# Returns: np array of estimated QPS (nx1)
def estimate_qps(X: IVEstimatorDataset, S: float, delta: int, ML_onnx: str, seed: int = None,
                 types: Tuple[np.dtype, np.dtype] = (None, None), input_type: int = 1,
                 input_names: Tuple[str, str]=("c_inputs", "d_inputs")):
    X_c = X.X_c
    X_d = X.X_d

    # If types not given, then infer from data
    types = list(types)
    if types[0] is None:
        types[0] = X.X_c.dtype
    if types[1] is None:
        if X_d is not None:
            types[1] = X.X_d.dtype

    # === Standardize continuous variables ===
    # Formula: (X_ik - u_k)/o_k; k represents a continuous variable
    mu = np.mean(X_c, axis=0)
    sigma = np.std(X_c, axis=0)
    X_c = np.apply_along_axis(lambda row: np.divide(np.subtract(row, mu), sigma), axis=1, arr=X_c)

    # TODO: parallelize the QPS estimation across individuals (seems like the most natural way to do it)
    #   - How does this work with GPU? Can we somehow partition the parallel jobs on the same GPU.
    if seed is not None:
        np.random.seed(seed)
    sess = rt.InferenceSession(ML_onnx)
    QPS_vec = []
    for i in range(X_c.shape[0]):
        if X_d is None:
            QPS_vec.append(_computeQPS(X_c[i,], types, S, delta, mu, sigma, sess, input_type, input_names=input_names)) # Compute QPS for each individual i
        else:
            QPS_vec.append(_computeQPS(X_c[i,], types, S, delta, mu, sigma, sess, X_di = X_d[i,], input_type=input_type, input_names=input_names)) # Compute QPS for each individual i
    QPS_vec = np.array(QPS_vec)
    return QPS_vec
