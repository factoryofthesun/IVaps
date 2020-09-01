# Testing the dataset Class
import sys
import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from mlisne.dataset import EstimatorDataset



def test_empty_initialization():
    dt = EstimatorDataset()
    attr = ['Y', 'Z', 'D', 'X_c', 'X_d']
    for key in attr:
        assert dt.__dict__[key] is None

def test_np_inference_initialization():
    cts_np = np.array(["Y", "Z", "D", 1, 2, 3, 4, 5])
    cts_np = np.tile(cts_np, (20,1))
    dt = EstimatorDataset(cts_np)
    assert all(dt.Y == np.array(["Y"]*20))
    assert all(dt.Z == np.array(["Z"]*20))
    assert all(dt.D == np.array(["D"]*20))
    assert np.array_equal(dt.X_c, np.tile(np.array([1,2,3,4,5]), (20,1)))
    assert dt.X_d is None

def test_np_index_initialization():
    cts_np = np.array([1, "Y", 2, "Z", 3, "D", 4, 5])
    cts_np = np.tile(cts_np, (20,1))
    C = [0,2,4,6]
    dt = EstimatorDataset(cts_np, X_c = C)
    assert all(dt.Y == np.array(["Y"]*20))
    assert all(dt.Z == np.array(["Z"]*20))
    assert all(dt.D == np.array(["D"]*20))
    assert np.array_equal(dt.X_c, np.tile(np.array([1,2,3,4]), (20,1)))
    assert np.array_equal(dt.X_d, np.array([5]*20))

def test_df_inference_initialization():
    cts_df = pd.DataFrame({"Y":["Y"]*20, "Z":["Z"]*20, "D":["D"]*20, "1":[1]*20,"2":[2]*20, "3":[3]*20})
    dt = EstimatorDataset(cts_df)
    assert all(dt.Y == np.array(["Y"]*20))
    assert all(dt.Z == np.array(["Z"]*20))
    assert all(dt.D == np.array(["D"]*20))
    assert np.array_equal(dt.X_c, np.tile(np.array([1,2,3]), (20,1)))
    assert dt.X_d is None

def test_df_index_initialization():
    cts_df = pd.DataFrame({"Y":["Y"]*20, "Z":["Z"]*20, "D":["D"]*20, "1":[1]*20,"2":[2]*20, "3":[3]*20})
    C = (3,5)
    dt = EstimatorDataset(cts_df, X_c = C)
    assert all(dt.Y == np.array(["Y"]*20))
    assert all(dt.Z == np.array(["Z"]*20))
    assert all(dt.D == np.array(["D"]*20))
    assert np.array_equal(dt.X_c, np.tile(np.array([1,3]), (20,1)))
    assert np.array_equal(dt.X_d, np.array([2]*20))

def test_str_initialization():
    sample_data_path = Path(__file__).resolve().parents[1] / "examples" / "data" / "iris_data.csv"
    C = range(4,8)
    dt = EstimatorDataset(sample_data_path, Y=8, Z = 2, D = 3, X_c = C)

    validate_df = pd.read_csv(sample_data_path)
    assert all(dt.Y == np.array(validate_df['Y']))
    assert all(dt.Z == np.array(validate_df['Z']))
    assert all(dt.D == np.array(validate_df["D"]))
    assert np.array_equal(dt.X_c, np.array(validate_df.iloc[:,C]))
    assert np.array_equal(dt.X_d, np.array(validate_df.iloc[:,0:2]))

def test_overwrite():
    cts_np = np.array(["Y", "Z", "D", 1, 2, 3, 4, 5])
    cts_np = np.tile(cts_np, (20,1))
    dt = EstimatorDataset(cts_np)

    Y_new = np.array(["Y_new"]*20)
    Z_new = pd.Series(["Z_new"]*20)
    D_new = ["D_new"]*20
    Xd_new = pd.DataFrame(np.tile(np.array(range(1,6)), (20,1)))
    dt.load_data(Y=Y_new, Z=Z_new, D=D_new, X_d=Xd_new)

    assert all(dt.Y == np.array(["Y_new"]*20))
    assert all(dt.Z == np.array(["Z_new"]*20))
    assert all(dt.D == np.array(["D_new"]*20))
    assert np.array_equal(dt.X_c, np.tile(np.array([1,2,3,4,5]), (20,1)))
    assert np.array_equal(dt.X_d, np.tile(np.array([1,2,3,4,5]), (20,1)))

def test_mixed_variables():
    sample_data_path = Path(__file__).resolve().parents[1] / "examples" / "data" / "iris_data.csv"
    C = range(4,8)
    L = {1:{3,4}}
    dt = EstimatorDataset(sample_data_path, Y=8, Z = 2, D = 3, X_c = C, L=L)

    validate_df = pd.read_csv(sample_data_path)
    assert all(dt.Y == np.array(validate_df['Y']))
    assert all(dt.Z == np.array(validate_df['Z']))
    assert all(dt.D == np.array(validate_df["D"]))
    assert np.array_equal(dt.X_c, np.array(validate_df.iloc[:,C]))
    assert np.array_equal(dt.X_d, np.array(validate_df.iloc[:,0:2]))
    assert dt.L == L

def test_mixed_value_error():
    sample_data_path = Path(__file__).resolve().parents[1] / "examples" / "data" / "iris_data.csv"
    C = range(4,8)
    L = {4:{3,4}}
    with pytest.raises(ValueError):
        dt = EstimatorDataset(sample_data_path, Y=8, Z = 2, D = 3, X_c = C, L=L)
