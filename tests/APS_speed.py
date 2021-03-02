# APS speed tests
import sys
import os
import pandas as pd
import numpy as np
import pytest
import pickle
from pathlib import Path
import multiprocessing
import pathos
import time
from sklearn.datasets import load_iris
import onnxruntime as rt
from pathlib import Path

from IVaps import estimate_aps_onnx, estimate_aps_user_defined
from IVaps.aps import _get_og_order

# First test that it works
def test_numba():
    sklearn_logreg = str(Path(__file__).resolve().parents[0] / "test_models" / "logreg_iris.onnx")
    iris = load_iris()
    X, y = iris.data, iris.target
    data = np.append(y[:,np.newaxis], X, axis = 1)
    dt = pd.DataFrame(data, columns = ["y", "x1", "x2", "x3", "x4"])
    data = dt.drop("y", axis = 1)
    data = pd.concat([data]*100, ignore_index = True) # 150k rows
    X_c = data.iloc[:,:3]
    X_d = data.iloc[:,3]

    # Insert NA into random rows
    random_rows = np.random.choice(np.arange(X_c.shape[0]), size=100)
    random_cols = np.random.choice(np.arange(X_c.shape[1]), size=100)

    for i in range(len(random_rows)):
        X_c.iloc[random_rows[i], random_cols[i]] = np.nan

    # Set second continuous variable as mixed
    L = {1:{3.0,4.0}}

    t0 = time.time()
    aps1 = estimate_aps_onnx_numba(X_c = X_c, X_d = X_d, S=100, delta=[0.1,0.5,0.8], onnx=sklearn_logreg, types=(np.float32,None), L = L, parallel=True)
    t1 = time.time()
    print("Numba runtime:", t1-t0)
    aps2 = estimate_aps_onnx(X_c = X_c, X_d = X_d, S=100, delta=[0.1,0.5,0.8], onnx=sklearn_logreg, types=(np.float32,None), L = L, parallel=True)
    t2 = time.time()
    print("Original APS runtime:", t2-t1)
    print(aps1)
    print(aps2)


if __name__ == "__main__":
    # Timeit
    from timeit import Timer

    sklearn_logreg = str(Path(__file__).resolve().parents[0] / "test_models" / "logreg_iris.onnx")
    iris = load_iris()
    X, y = iris.data, iris.target
    data = np.append(y[:,np.newaxis], X, axis = 1)
    dt = pd.DataFrame(data, columns = ["y", "x1", "x2", "x3", "x4"])
    data = dt.drop("y", axis = 1)

    # mult_vals = [1, 50, 100, 500, 1000, 2500, 5000]
    # N = [150 * mult for mult in mult_vals]
    # numba_time = []
    # og_time = []
    # for mult in mult_vals:
    #     tmp = pd.concat([data]*mult, ignore_index = True) # 150k rows
    #     X_c = tmp.iloc[:,:3]
    #     X_d = tmp.iloc[:,3]
    #
    #     # Insert NA into random rows
    #     random_rows = np.random.choice(np.arange(X_c.shape[0]), size=100)
    #     random_cols = np.random.choice(np.arange(X_c.shape[1]), size=100)
    #     for i in range(len(random_rows)):
    #         X_c.iloc[random_rows[i], random_cols[i]] = np.nan
    #
    #     # Set second continuous variable as mixed
    #     L = {1:{3.0,4.0}}
    #
    #     # Time functions
    #     t = Timer(lambda: estimate_aps_onnx_numba(X_c = X_c, X_d = X_d, S=100, delta=[0.1,0.5,0.8],
    #                 onnx=sklearn_logreg, types=(np.float32,None), L = L, parallel=True))
    #     numba_runtime = t.timeit(5)/300
    #     numba_time.append(numba_runtime)
    #     t = Timer(lambda: estimate_aps_onnx(X_c = X_c, X_d = X_d, S=100, delta=[0.1,0.5,0.8],
    #                 onnx=sklearn_logreg, types=(np.float32,None), L = L, parallel=True))
    #     og_runtime = t.timeit(5)/300
    #     og_time.append(og_runtime)
    #
    # import matplotlib.pyplot as plt
    # plt.plot(N, numba_time, color="red", label="Numba APS Performance")
    # plt.plot(N, og_time, color="black", label="Original APS Performance")
    # plt.legend()
    # plt.title("Comparing APS Performance")
    # plt.savefig("speed_test.png")
    # plt.show()
    # L = {1:{3.0,4.0}}
    # X_c = data.iloc[:,:3]
    # X_d = data.iloc[:,3]
    # aps = estimate_aps_onnx(X_c = X_c, X_d = X_d, S=100, delta=[0.1,0.5,0.8], onnx=sklearn_logreg, types=(np.float32,None))
    # print("APS ---------")
    # print(aps)
    # aps_p = estimate_aps_onnx(X_c = X_c, X_d = X_d, S=100, delta=[0.1,0.5,0.8], onnx=sklearn_logreg, types=(np.float32,None), parallel=True)
    # print("APS parallel ---------")
    # print(aps_p)
    # aps_l = estimate_aps_onnx(X_c = X_c, X_d = X_d, S=100, delta=[0.1,0.5,0.8], onnx=sklearn_logreg, types=(np.float32,None), L = L, parallel=True)
    # print("APS discrete vals ---------")
    # print(aps_l)
