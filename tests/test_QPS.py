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

from mlisne.dataset import EstimatorDataset
from mlisne.qps import estimate_qps, estimate_qps_user_defined

sklearn_logreg = str(Path(__file__).resolve().parents[0] / "test_models" / "logreg_iris.onnx")
sklearn_logreg_double = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_double.onnx")
sklearn_logreg_infer = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_infertype.onnx")
model_path = str(Path(__file__).resolve().parents[1] / "examples" / "models")

@pytest.fixture
def iris_dataset():
    """Returns EstimatorDataset with iris loaded"""
    iris = load_iris()
    X, y = iris.data, iris.target
    dt = EstimatorDataset(Y = y, X_c = X)
    return dt

@pytest.fixture
def iris_dataset_discrete():
    """Returns EstimatorDataset with iris loaded and last column set as discrete"""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_c = X[:,:3]
    X_d = X[:,3]
    dt = EstimatorDataset(Y = y, X_c = X_c, X_d = X_d)
    return dt

def test_estimate_nodiscrete_skl(iris_dataset):
    qps = estimate_qps(iris_dataset, S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None))
    print(qps)
    assert qps.shape[0] == iris_dataset.Y.shape[0]

def test_seed_skl(iris_dataset):
    seed = np.random.choice(range(100))
    qps1 = estimate_qps(iris_dataset, S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    qps2 = estimate_qps(iris_dataset, S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    assert np.array_equal(qps1, qps2)

def test_estimate_withdiscrete_skl(iris_dataset_discrete):
    qps = estimate_qps(iris_dataset_discrete, S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None))
    print(qps)
    assert qps.shape[0] == iris_dataset_discrete.Y.shape[0]

def test_estimate_double_skl(iris_dataset_discrete):
    qps = estimate_qps(iris_dataset_discrete, S=100, delta=0.8, ML_onnx=sklearn_logreg_double, types=(np.float64, None))
    print(qps)
    assert qps.shape[0] == iris_dataset_discrete.Y.shape[0]

def test_estimate_infer_skl(iris_dataset_discrete):
    iris_dataset_discrete.X_c = iris_dataset_discrete.X_c.astype(np.float32)
    iris_dataset_discrete.X_d = iris_dataset_discrete.X_d.astype(np.float32)
    qps = estimate_qps(iris_dataset_discrete, S=100, delta=0.8, ML_onnx=sklearn_logreg_infer)
    print(qps)
    assert qps.shape[0] == iris_dataset_discrete.Y.shape[0]

def test_decision_function_round(iris_dataset_discrete):
    iris_dataset_discrete.X_c = iris_dataset_discrete.X_c.astype(np.float32)
    iris_dataset_discrete.X_d = iris_dataset_discrete.X_d.astype(np.float32)
    seed = np.random.choice(range(100))
    qps_round = estimate_qps(iris_dataset_discrete, ML_onnx=sklearn_logreg_infer, seed = seed, fcn=round)
    qps_round_vec = estimate_qps(iris_dataset_discrete, ML_onnx=sklearn_logreg_infer, seed = seed, fcn=np.round,
                                                        vectorized=True)
    assert np.array_equal(qps_round, qps_round_vec)

def test_user_ml_function(iris_dataset_discrete):
    model = pickle.load(open(f"{model_path}/iris_logreg.pickle", 'rb'))

    # Basic decision function: assign treatment if prediction > c
    def assign_cutoff(X, c):
        return (X > c).astype("int")

    # User-defined ML function
    def ml_round(X, **kwargs):
        preds = model.predict_proba(X)
        treat = assign_cutoff(preds, **kwargs)
        return treat

    qps = estimate_qps_user_defined(iris_dataset_discrete, ml_round, c = 0.5)
    qps0 = estimate_qps_user_defined(iris_dataset_discrete, ml_round, c = 1)
    qps1 = estimate_qps_user_defined(iris_dataset_discrete, ml_round, c = 0)
    print(qps)

    assert qps.shape[0] == iris_dataset_discrete.Y.shape[0]
    assert np.all(qps0 == 0)
    assert np.all(qps1 == 1)

def test_cts_nan(iris_dataset_discrete):
    seed = np.random.choice(range(100))

    # Insert NA into random rows
    random_rows = np.random.choice(np.arange(iris_dataset_discrete.X_c.shape[0]), size=100)
    random_cols = np.random.choice(np.arange(iris_dataset_discrete.X_c.shape[1]), size=100)
    iris_dataset_discrete.X_c[random_rows, random_cols] = np.nan
    print(iris_dataset_discrete.X_c)

    qps1 = estimate_qps(iris_dataset_discrete, S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    qps2 = estimate_qps(iris_dataset_discrete, S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    np.testing.assert_array_equal(qps1, qps2)

def test_iris_mixed_vars(iris_dataset_discrete):
    seed = np.random.choice(range(100))

    # Set second continuous variable as mixed
    L = {1:{3.0,4.0}}
    iris_dataset_discrete.L = L
    print(iris_dataset_discrete.X_c[:,1])

    qps1 = estimate_qps(iris_dataset_discrete, S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    qps2 = estimate_qps(iris_dataset_discrete, S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    np.testing.assert_array_equal(qps1, qps2)
