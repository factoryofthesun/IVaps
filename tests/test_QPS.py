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
from mlisne.qps import _get_og_order

sklearn_logreg = str(Path(__file__).resolve().parents[0] / "test_models" / "logreg_iris.onnx")
sklearn_logreg_double = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_double.onnx")
sklearn_logreg_infer = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_infertype.onnx")
model_path = str(Path(__file__).resolve().parents[1] / "examples" / "models")

@pytest.fixture
def iris_dataset():
    """Returns EstimatorDataset with iris loaded"""
    iris = load_iris()
    X, y = iris.data, iris.target
    data = np.append(y[:,np.newaxis], X, axis = 1)
    dt = pd.DataFrame(data, columns = ["y", "x1", "x2", "x3", "x4"])
    return dt

def test_inp_error(iris_dataset):
    with pytest.raises(ValueError):
        qps = estimate_qps_onnx(X_d = iris_dataset, D = 1, C = 10, S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None))

    def dummy(X):
        return X

    with pytest.raises(ValueError):
        qps = estimate_qps_user_defined(ml = dummy, X_d = iris_dataset, D = 12, C = 1, S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None))

def test_estimate_nodiscrete_skl(iris_dataset):
    qps = estimate_qps_onnx(X_c = iris_dataset.iloc[:,1:], S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None))
    print(qps)
    assert qps.shape[0] == iris_dataset.shape[0]

def test_estimate_nodiscrete_index_skl(iris_dataset):
    qps = estimate_qps_onnx(data = iris_dataset.drop("y", axis=1), C = range(4), S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None))
    print(qps)
    assert qps.shape[0] == iris_dataset.shape[0]

def test_estimate_nodiscrete_infer_skl(iris_dataset):
    qps = estimate_qps_onnx(data = iris_dataset.drop("y", axis=1), S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None))
    print(qps)
    assert qps.shape[0] == iris_dataset.shape[0]

def test_seed_skl(iris_dataset):
    seed = np.random.choice(range(100))
    qps1 = estimate_qps_onnx(data = iris_dataset.drop("y", axis=1), S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    qps2 = estimate_qps_onnx(data = iris_dataset.drop("y", axis=1), S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    assert np.array_equal(qps1, qps2)

def test_estimate_withdiscrete_skl(iris_dataset):
    qps = estimate_qps_onnx(X_c = iris_dataset.iloc[:,1:4], X_d = iris_dataset.iloc[:,4], data = iris_dataset, S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None), seed = 1)
    qps2 = estimate_qps_onnx(data = iris_dataset, C = range(1,4), D = 4, S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None), seed = 1)

    assert np.array_equal(qps, qps2)

def test_estimate_withdiscrete_infer_disc_skl(iris_dataset):
    qps1 = estimate_qps_onnx(data = iris_dataset.drop("y", axis = 1), C = range(3), S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None), seed = 1)
    qps2 = estimate_qps_onnx(data = iris_dataset["x4"], X_c = iris_dataset[["x1", "x2", "x3"]], S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None), seed = 1)
    assert np.array_equal(qps1, qps2)

def test_estimate_withdiscrete_infer_cts_skl(iris_dataset):
    qps1 = estimate_qps_onnx(data = iris_dataset.drop("y", axis = 1), D = 3, S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None), seed = 1)
    qps2 = estimate_qps_onnx(data = iris_dataset.iloc[:,1:4], X_d = iris_dataset["x4"], S=100, delta=0.8, ML_onnx=sklearn_logreg, types=(np.float32,None), seed = 1)
    assert np.array_equal(qps1, qps2)

def test_estimate_double_skl(iris_dataset):
    qps = estimate_qps_onnx(data = iris_dataset.drop("y", axis = 1), S=100, delta=0.8, ML_onnx=sklearn_logreg_double, types=(np.float64, None))
    print(qps)
    assert qps.shape[0] == iris_dataset.shape[0]

def test_estimate_infer_skl(iris_dataset):
    data = iris_dataset.astype(np.float32)
    qps = estimate_qps_onnx(data = data.drop("y", axis = 1), S=100, delta=0.8, ML_onnx=sklearn_logreg_infer)
    print(qps)
    assert qps.shape[0] == iris_dataset.shape[0]

def test_decision_function_round(iris_dataset):
    data = iris_dataset.astype(np.float32)
    seed = np.random.choice(range(100))
    qps_round = estimate_qps_onnx(data = data.drop("y", axis = 1), ML_onnx=sklearn_logreg_infer, seed = seed, fcn=round)
    qps_round_vec = estimate_qps_onnx(data = data.drop("y", axis = 1), ML_onnx=sklearn_logreg_infer, seed = seed, fcn=np.round,
                                                        vectorized=True)
    assert np.array_equal(qps_round, qps_round_vec)

def test_user_ml_function(iris_dataset):
    model = pickle.load(open(f"{model_path}/iris_logreg.pickle", 'rb'))
    data = iris_dataset.drop("y", axis = 1)

    # Basic decision function: assign treatment if prediction > c
    def assign_cutoff(X, c):
        return (X > c).astype("int")

    # User-defined ML function
    def ml_round(X, **kwargs):
        preds = model.predict_proba(X)
        treat = assign_cutoff(preds, **kwargs)
        return treat

    qps = estimate_qps_user_defined(ml_round, data = data, C = range(3), D = 3, c = 0.5)
    qps0 = estimate_qps_user_defined(ml_round, data = data, C = range(3), c = 1)
    qps1 = estimate_qps_user_defined(ml_round, data = data, D = 3, c = 0)
    print(qps)

    assert qps.shape[0] == iris_dataset.shape[0]
    assert np.all(qps0 == 0)
    assert np.all(qps1 == 1)

def test_user_pandas(iris_dataset):
    model = pickle.load(open(f"{model_path}/iris_logreg.pickle", 'rb'))
    data = iris_dataset.drop("y", axis = 1)

    # Basic decision function: assign treatment if prediction > c
    def assign_cutoff(X, c):
        return (X > c).astype("int")

    # User-defined ML function
    def ml_round(X, **kwargs):
        print(f"Pandas columns: {X.columns}")
        X = X.to_numpy()
        preds = model.predict_proba(X)
        treat = assign_cutoff(preds, **kwargs)
        return treat

    qps = estimate_qps_user_defined(ml_round, data = data.iloc[:,3], X_c = data.iloc[:,:3], c = 0.5, pandas = True, pandas_cols = ['C1', 'C2', 'C3', 'C4'])
    qps0 = estimate_qps_user_defined(ml_round, data = data.iloc[:,:3], X_d = data['x4'], c = 1, pandas = True)
    qps1 = estimate_qps_user_defined(ml_round, X_c = data.iloc[:,:3], X_d = data.iloc[:,3], c = 0, pandas = True, pandas_cols = ['T1', 'T2', 'T3', 'T4'])
    print(qps)

    assert qps.shape[0] == iris_dataset.shape[0]
    assert np.all(qps0 == 0)
    assert np.all(qps1 == 1)

def test_cts_nan(iris_dataset):
    seed = np.random.choice(range(100))

    X_c = iris_dataset.iloc[:,1:4]

    # Insert NA into random rows
    random_rows = np.random.choice(np.arange(X_c.shape[0]), size=100)
    random_cols = np.random.choice(np.arange(X_c.shape[1]), size=100)
    X_c.iloc[random_rows, random_cols] = np.nan
    print(X_c)

    qps1 = estimate_qps_onnx(X_c = X_c, X_d = iris_dataset.iloc[:,4], S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    qps2 = estimate_qps_onnx(X_c = X_c, X_d = iris_dataset.iloc[:,4], S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    np.testing.assert_array_equal(qps1, qps2)

def test_iris_mixed_vars(iris_dataset):
    seed = np.random.choice(range(100))

    # Set second continuous variable as mixed
    L = {1:{3.0,4.0}}
    print(iris_dataset.iloc[:,1])

    qps1 = estimate_qps_onnx(data = iris_dataset.drop('y', axis = 1), S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    qps2 = estimate_qps_onnx(data = iris_dataset.drop('y', axis = 1), S=100, delta=0.8, ML_onnx=sklearn_logreg, seed = seed, types=(np.float32,None))
    np.testing.assert_array_equal(qps1, qps2)

def test_1d(iris_dataset):
    seed = np.random.choice(range(100))

    # Basic decision function: assign treatment if prediction > c
    def assign_cutoff(X, c):
        return (X > c).astype("int")

    # User-defined ML function
    def ml_round(X, **kwargs):
        assert X.shape[1] == 1
        X[(X < 0) | (X > 1)] = 0.5
        treat = assign_cutoff(X, **kwargs)
        return treat

    qps0 = estimate_qps_user_defined(ml_round, X_c = iris_dataset['x1'], c = 1)
    qps1 = estimate_qps_user_defined(ml_round, X_c = iris_dataset['x1'], c = 0)

    assert np.all(qps0 == 0)
    assert np.all(qps1 == 1)

def test_pandas_keep_order(iris_dataset):
    # User defined function that is dependent on column order
    def iris_order(X, **kwargs):
        assert all(X.columns == ['x1', 'x2', 'x3', 'x4'])
        return np.array([1]*len(X))

    qps = estimate_qps_user_defined(iris_order, data = iris_dataset.iloc[:,1:], C = [2,3], D = [0,1],
                                    pandas = True, keep_order = True, pandas_cols = ['x1', 'x2', 'x3', 'x4'])

    order = _get_og_order(4, [2,3], [0,1])
    assert order == [2,3,0,1]

def test_pandas_reorder(iris_dataset):
    seed = np.random.choice(range(100))

    # User defined function that is dependent on column order
    def iris_order(X, **kwargs):
        assert all(X.columns == ['x3', 'x4', 'x1', 'x2'])
        return X.iloc[:,0] * 0.5

    qps = estimate_qps_user_defined(iris_order, data = iris_dataset.iloc[:,1:], C = [2,3], D = [0,1],
                                    pandas = True, keep_order = True, reorder = [2,3,0,1],
                                    pandas_cols = ['x3', 'x4', 'x1', 'x2'], seed = seed)
    qps2 = estimate_qps_user_defined(iris_order, data = iris_dataset.iloc[:,1:], C = [2,3], D = [0,1],
                                    pandas = True, pandas_cols = ['x3', 'x4', 'x1', 'x2'], seed = seed)
    assert np.array_equal(qps, qps2)
