# Test QPS estimation
import sys
import os
import pandas as pd
import numpy as np
import pytest
import pickle
from pathlib import Path
from sklearn.datasets import load_iris
import onnxruntime as rt
from pathlib import Path

from mlisne import estimate_qps_onnx, estimate_qps_user_defined
from mlisne.qps_old import _estimate_qps_onnx, _estimate_qps_user_defined
from mlisne.qps import _get_og_order

sklearn_logreg = str(Path(__file__).resolve().parents[0] / "test_models" / "logreg_iris.onnx")
sklearn_logreg_double = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_double.onnx")
sklearn_logreg_infer = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_infertype.onnx")
model_path = str(Path(__file__).resolve().parents[1] / "examples" / "models")

L = {1:{3.0,4.0}}
L_keys = np.array(list(L.keys()))
L_vals = np.array(list(L.values()))

iris = load_iris()
X, y = iris.data, iris.target
data = np.append(y[:,np.newaxis], X, axis = 1)
X_c = np.array(pd.DataFrame(data, columns = ["y", "x1", "x2", "x3", "x4"]).drop("y", axis = 1))

# Get indices of mixed vars to replace for each row
mixed_og_rows = [np.where(np.isin(X_c[:,L_keys[i]], list(L_vals[i])))[0] for i in range(len(L_keys))] # List of row indices for each mixed variable column
mixed_og_cols = [np.repeat(L_keys[i], len(mixed_og_rows[i])) for i in range(len(mixed_og_rows))]
mixed_og_inds = (np.concatenate(mixed_og_rows), np.concatenate(mixed_og_cols))

# Save original discrete values
mixed_og_vals = X_c[mixed_og_inds]

# Replace values at indices with NA
X_c[mixed_og_inds] = np.nan

print(X_c)
long_X = np.repeat(X_c, 3, axis = 0).reshape(-1,3,X_c.shape[1])

print(long_X)
long_X[mixed_og_inds[0],:, mixed_og_inds[1]] = mixed_og_vals[:,np.newaxis]
print(long_X[mixed_og_inds[0],:,:])
