# Test QPS estimation
import sys
import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from sklearn.datasets import load_iris
import onnxruntime as rt
from pathlib import Path

from mlisne.dataset import IVEstimatorDataset
from mlisne.helpers import estimate_qps

sklearn_logreg = str(Path(__file__).resolve().parents[1] / "examples" / "models" / "logreg_iris.onnx")
sklearn_logreg_double = str(Path(__file__).resolve().parents[1] / "examples" / "models" / "logreg_iris_double.onnx")
sklearn_logreg_infer = str(Path(__file__).resolve().parents[1] / "examples" / "models" / "logreg_iris_infertype.onnx")

@pytest.fixture
def iris_dataset():
    """Returns IVEstimatorDataset with iris loaded"""
    iris = load_iris()
    X, y = iris.data, iris.target
    dt = IVEstimatorDataset(Y = y, X_c = X)
    return dt

@pytest.fixture
def iris_dataset_discrete():
    """Returns IVEstimatorDataset with iris loaded and last column set as discrete"""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_c = X[:,:3]
    X_d = X[:,3]
    dt = IVEstimatorDataset(Y = y, X_c = X_c, X_d = X_d)
    return dt

def test_estimate_nodiscrete_skl(iris_dataset):
    qps = estimate_qps(iris_dataset.X_c, S=100, delta=0.8, ML_onnx=sklearn_logreg)
    print(qps)
    assert qps.shape[0] == iris_dataset.Y.shape[0]

def test_seed_skl(iris_dataset):
    seed = np.random.choice(range(100))
    qps1 = estimate_qps(iris_dataset.X_c, S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed)
    qps2 = estimate_qps(iris_dataset.X_c, S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed)
    assert np.array_equal(qps1, qps2)

def test_estimate_withdiscrete_skl(iris_dataset_discrete):
    qps = estimate_qps(iris_dataset_discrete.X_c, X_d = iris_dataset_discrete.X_d, S=100, delta=0.8, ML_onnx=sklearn_logreg)
    print(qps)
    assert qps.shape[0] == iris_dataset_discrete.Y.shape[0]

def test_estimate_double_skl(iris_dataset_discrete):
    qps = estimate_qps(iris_dataset_discrete.X_c, X_d = iris_dataset_discrete.X_d,
                       S=100, delta=0.8, ML_onnx=sklearn_logreg_double, type=np.float64)
    print(qps)
    assert qps.shape[0] == iris_dataset_discrete.Y.shape[0]

def test_estimate_infer_skl(iris_dataset_discrete):
    qps = estimate_qps(iris_dataset_discrete.X_c, X_d = iris_dataset_discrete.X_d,
                       S=100, delta=0.8, ML_onnx=sklearn_logreg_infer)
    print(qps)
    assert qps.shape[0] == iris_dataset_discrete.Y.shape[0]
