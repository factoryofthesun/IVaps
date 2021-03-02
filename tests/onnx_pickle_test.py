# Test APS estimation
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

from IVaps import estimate_aps_onnx, estimate_aps_user_defined
from IVaps.aps import _get_og_order

iris_onnx = str(Path(__file__).resolve().parents[0] / "test_models" / "iris_test.onnx")
sklearn_logreg = str(Path(__file__).resolve().parents[0] / "test_models" / "logreg_iris.onnx")
sklearn_logreg_double = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_double.onnx")
sklearn_logreg_infer = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_infertype.onnx")
model_path = str(Path(__file__).resolve().parents[1] / "examples" / "models")
data_path = str(Path(__file__).resolve().parents[1] / "examples" / "data")

def create_and_run_session(X):
    if X.ndim == 1:
        X = X[np.newaxis,:]
    sess = rt.InferenceSession(iris_onnx)
    input_name = sess.get_inputs()[0].name
    preds = sess.run(None, {input_name: X})
    return True

if __name__ == "__main__":
    iris = pd.read_csv(f"{data_path}/iris_data.csv")
    X_inp = np.array(iris[['X1', 'X2', 'X3', 'X4']])

    import pathos
    mp = pathos.helpers.mp
    p = mp.Pool()
    res = p.map(create_and_run_session, iter(X_inp))
