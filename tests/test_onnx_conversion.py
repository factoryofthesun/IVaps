# Test ONNX conversion
import sys
import os
import pandas as pd
import numpy as np
import pickle
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.datasets import load_iris
import onnxruntime as rt
from pathlib import Path

from mlisne.helpers import convert_to_onnx

model_path = str(Path(__file__).resolve().parents[1] / "examples" / "models")
data_path = str(Path(__file__).resolve().parents[1] / "examples" / "data")

# Model without categorical embeddings
class Model(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.4):
        super().__init__()

        all_layers=[]
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

        self.m = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.m(x)

        return x[:,1]

# Model with categorical embeddings
class CatModel(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

        self.m = nn.Softmax(dim=1)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        x = self.m(x)

        return x[:,1]

@pytest.fixture
def iris_model():
    model = pickle.load(open(f"{model_path}/iris_logreg.pickle", 'rb'))
    return model

@pytest.fixture
def churn_model():
    model = Model(10, 2, [200,100,50], p=0.4)
    model.load_state_dict(torch.load(f"{model_path}/churn.pt"))
    model.eval()
    return model

@pytest.fixture
def churn_cat_model():
    model = CatModel([(3, 2), (2, 1), (2, 1), (2, 1)], 6, 2, [200,100,50], p=0.4)
    model.load_state_dict(torch.load(f"{model_path}/churn_categorical.pt"))
    model.eval()
    return model

@pytest.fixture
def iris_data():
    iris = pd.read_csv(f"{data_path}/iris_data.csv")
    return iris

@pytest.fixture
def churn_data():
    churn = pd.read_csv(f"{data_path}/churn_data.csv")
    return churn

def test_sklearn_conversion(iris_model, iris_data):
    X_dummy = np.array(iris_data.loc[0, ['X_0', 'X_1', 'X_2', 'X_3']])
    X_inp = np.array(iris_data[['X_0', 'X_1', 'X_2', 'X_3']])
    f = "test_models/iris_test.onnx"
    convert_to_onnx(iris_model, X_dummy, f, "sklearn")

    # Test inference
    skl_preds = iris_model.predict_proba(X_inp)
    sess = rt.InferenceSession(f)
    label_name = 'output_probability'
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run([label_name], {input_name: X_inp})[0]
    onnx_preds_np = np.array([[d[0], d[1]] for d in onnx_preds])

    np.testing.assert_array_almost_equal(skl_preds, onnx_preds_np, decimal=5)

def test_torch_conversion(churn_model, churn_data):
    categorical_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    for category in categorical_cols:
        churn_data[category] = churn_data[category].astype('category')
    cat = []
    for c in categorical_cols:
        cat.append(churn_data[c].cat.codes.values)
    cat_data = np.stack(cat, 1)
    num_data = np.array(churn_data[numerical_cols])
    tot_data = np.concatenate((cat_data, num_data), axis=1)

    cat_tensor = torch.tensor(cat_data).double()
    num_tensor = torch.tensor(num_data)
    tot_tensor = torch.cat((cat_tensor, num_tensor), 1).float()
    X_dummy = tot_tensor[0,None]

    f = "test_models/churn_test.onnx"
    convert_to_onnx(churn_model, X_dummy, f, "pytorch")

    # Test inference
    with torch.no_grad():
        torch_preds = churn_model(tot_tensor).numpy()
    sess = rt.InferenceSession(f)
    onnx_preds = sess.run(["output_probability"], {"input": tot_data.astype(np.float32)})[0]

    np.testing.assert_array_almost_equal(torch_preds, onnx_preds, decimal=5)
