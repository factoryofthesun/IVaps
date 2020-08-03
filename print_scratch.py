# Test IV Estimator
import sys
import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from sklearn.datasets import load_iris
import onnxruntime as rt
from pathlib import Path
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

from mlisne.dataset import IVEstimatorDataset
from mlisne.helpers import estimate_qps
from mlisne.estimator import TreatmentIVEstimator

data_path = "D:\Tobin\mlisne\examples\data"
model_path = "D:\Tobin\mlisne\examples\models"

iris_data = iris = pd.read_csv(f"{data_path}/iris_data.csv")
empty_estimator = TreatmentIVEstimator()

qps = np.array(iris_data['QPS'])
data = np.array(iris_data.drop("QPS", axis=1))
dataset = IVEstimatorDataset(data)
empty_estimator.fit(dataset, qps)

print(empty_estimator)
