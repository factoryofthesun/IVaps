# Test ONNX conversion
import sys
import os
import pandas as pd
import numpy as np
import pickle
import pytest
from pathlib import Path
from sklearn.datasets import load_iris
import onnxruntime as rt
from pathlib import Path
import torch
import torch.nn as nn
from IVaps.helpers import convert_to_onnx

model_path = str(Path(__file__).resolve().parents[1] / "examples" / "models")
data_path = str(Path(__file__).resolve().parents[1] / "examples" / "data")

import pkg_resources
installed = {pkg.key for pkg in pkg_resources.working_set}

needs_tf = pytest.mark.skipif("tensorflow" not in installed,
                            reason = "Needs tensorflow.")

@pytest.fixture
def iris_model():
    model = pickle.load(open(f"{model_path}/iris_logreg.pickle", 'rb'))
    return model

@pytest.fixture
def churn_model():
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

    model = Model(10, 2, [200,100,50], p=0.4)
    model.load_state_dict(torch.load(f"{model_path}/churn.pt"))
    model.eval()
    return model

@pytest.fixture
def churn_cat_model():
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

    model = CatModel([(3, 2), (2, 1), (2, 1), (2, 1)], 6, 2, [200,100,50], p=0.4)
    model.load_state_dict(torch.load(f"{model_path}/churn_categorical.pt"))
    model.eval()
    return model

@pytest.fixture()
def lgbm_model():
    import lightgbm as lgb
    model = lgb.Booster(model_file= f"{model_path}/lgbm_example.txt")
    return model

@pytest.fixture()
def keras_model():
    import keras

    model = keras.models.load_model(f"{model_path}/keras_example")
    return model

@pytest.fixture
def iris_data():
    iris = pd.read_csv(f"{data_path}/iris_data.csv")
    return iris

@pytest.fixture
def churn_data():
    churn = pd.read_csv(f"{data_path}/churn_data.csv")
    return churn

@pytest.fixture
def lgbm_data():
    data = pd.read_csv(f"{data_path}/lgbm_regression.test", header=None, sep='\t')
    return data

@pytest.fixture
def ssd_data():
    import numpy as np
    from PIL import Image, ImageDraw, ImageColor
    import math

    img = Image.open(f"{data_path}/ssd_image.jpg")
    img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)

    return img_data

def test_sklearn_conversion(iris_model, iris_data):
    X_dummy = np.array(iris_data.loc[0, ['X1', 'X2', 'X3', 'X4']])
    X_inp = np.array(iris_data[['X1', 'X2', 'X3', 'X4']])
    f = os.path.join(os.path.dirname(__file__), "test_models/iris_test.onnx")
    convert_to_onnx(iris_model, "sklearn", X_dummy, output_path = f)

    # Test inference
    skl_preds = iris_model.predict_proba(X_inp)
    sess = rt.InferenceSession(f)
    label_name = 'output_probability'
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run([label_name], {input_name: X_inp})[0]
    onnx_preds_np = np.array([[d[0], d[1]] for d in onnx_preds])

    np.testing.assert_array_almost_equal(skl_preds, onnx_preds_np, decimal=5)

def test_sklearn_input_types(iris_model, iris_data):
    X_inp = np.array(iris_data[['X1', 'X2', 'X3', 'X4']])

    # Standard input
    X_dummy = np.array(iris_data.loc[0, ['X1', 'X2', 'X3', 'X4']])
    f = os.path.join(os.path.dirname(__file__), "test_models/iris_test.onnx")
    convert_to_onnx(iris_model, "sklearn", X_dummy, output_path = f)

    sess = rt.InferenceSession(f)
    label_name = 'output_probability'
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run([label_name], {input_name: X_inp})[0]
    onnx_preds0 = np.array([[d[0], d[1]] for d in onnx_preds])

    # Pandas 1 row input
    X_dummy = iris_data.loc[0, ['X1', 'X2', 'X3', 'X4']]
    f = os.path.join(os.path.dirname(__file__), "test_models/iris_pandas1_test.onnx")
    convert_to_onnx(iris_model, "sklearn", X_dummy, output_path = f)
    sess = rt.InferenceSession(f)

    label_name = 'output_probability'
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run([label_name], {input_name: X_inp})[0]
    onnx_preds1 = np.array([[d[0], d[1]] for d in onnx_preds])

    # Pandas 2-dim input
    X_dummy = iris_data[['X1', 'X2', 'X3', 'X4']]
    f = os.path.join(os.path.dirname(__file__), "test_models/iris_pandas2_test.onnx")
    convert_to_onnx(iris_model, "sklearn", X_dummy, output_path = f)

    sess = rt.InferenceSession(f)
    label_name = 'output_probability'
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run([label_name], {input_name: X_inp})[0]
    onnx_preds2 = np.array([[d[0], d[1]] for d in onnx_preds])

    # Pytorch tensor
    X_dummy = torch.tensor(np.array(iris_data.loc[0, ['X1', 'X2', 'X3', 'X4']]))
    f = os.path.join(os.path.dirname(__file__), "test_models/iris_torch_test.onnx")
    convert_to_onnx(iris_model, "sklearn", X_dummy, output_path = f)

    sess = rt.InferenceSession(f)
    label_name = 'output_probability'
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run([label_name], {input_name: X_inp})[0]
    onnx_preds3 = np.array([[d[0], d[1]] for d in onnx_preds])

    # Test inference
    np.testing.assert_array_almost_equal(onnx_preds0, onnx_preds1, decimal=5)
    np.testing.assert_array_almost_equal(onnx_preds1, onnx_preds2, decimal=5)
    np.testing.assert_array_almost_equal(onnx_preds2, onnx_preds3, decimal=5)

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
    tot_data = np.concatenate((cat_data, num_data), axis=1).astype(np.float32)

    cat_tensor = torch.tensor(cat_data).double()
    num_tensor = torch.tensor(num_data)
    tot_tensor = torch.cat((cat_tensor, num_tensor), 1).float()
    X_dummy = tot_tensor[0,None]

    f = os.path.join(os.path.dirname(__file__), "test_models/churn_test.onnx")
    convert_to_onnx(churn_model, "pytorch",X_dummy, output_path = f)

    f_notensor = os.path.join(os.path.dirname(__file__), "test_models/churn_test_notensor.onnx")
    convert_to_onnx(churn_model, "pytorch",tot_data, output_path = f_notensor)

    # Test inference
    with torch.no_grad():
        torch_preds = churn_model(tot_tensor).numpy()
    sess = rt.InferenceSession(f)
    output_name = sess.get_outputs()[0].name
    onnx_preds = sess.run([output_name], {"c_inputs": tot_data.astype(np.float32)})[0]

    sess = rt.InferenceSession(f_notensor)
    output_name = sess.get_outputs()[0].name
    onnx_preds_notensor = sess.run([output_name], {"c_inputs": tot_data.astype(np.float32)})[0]

    np.testing.assert_array_almost_equal(torch_preds, onnx_preds, decimal=5)
    np.testing.assert_array_almost_equal(onnx_preds, onnx_preds_notensor, decimal=5)

def test_torch_cat_conversion(churn_cat_model, churn_data):
    categorical_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    for category in categorical_cols:
        churn_data[category] = churn_data[category].astype('category')
    cat = []
    for c in categorical_cols:
        cat.append(churn_data[c].cat.codes.values)
    cat_data = np.stack(cat, 1)
    num_data = np.array(churn_data[numerical_cols])

    cat_tensor = torch.tensor(cat_data).long()
    num_tensor = torch.tensor(num_data).float()

    cat_dummy = cat_tensor[0,None]
    num_dummy = num_tensor[0,None]

    print(cat_dummy.size(), num_dummy.size())
    f = os.path.join(os.path.dirname(__file__), "test_models/churn_cat_test.onnx")
    convert_to_onnx(churn_cat_model, "pytorch", cat_dummy, num_dummy, output_path = f, input_names=("d_inputs", "c_inputs"))

    # Test inference
    with torch.no_grad():
        torch_preds = churn_cat_model(cat_tensor, num_tensor).numpy()
    sess = rt.InferenceSession(f)
    output_name = sess.get_outputs()[0].name
    onnx_preds = sess.run([output_name], {"c_inputs": num_data.astype(np.float32),
                                                   "d_inputs": cat_data.astype(np.int64)})[0]

    np.testing.assert_array_almost_equal(torch_preds, onnx_preds, decimal=5)

def test_lgbm_conversion(lgbm_model, lgbm_data):
    X = lgbm_data.drop(0, axis=1)
    og_preds = lgbm_model.predict(X)

    X_dummy = np.array(X)[0,:]
    print(X_dummy.dtype)

    f = os.path.join(os.path.dirname(__file__), "test_models/lgbm.onnx")
    convert_to_onnx(lgbm_model, "lightgbm", X_dummy, output_path = f, target_opset=12)

    sess = rt.InferenceSession(f)
    label_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run([label_name], {input_name: np.array(X)})[0]
    print(onnx_preds.shape)

    np.testing.assert_array_almost_equal(og_preds, np.squeeze(onnx_preds), decimal=5)

def test_keras_conversion(keras_model, iris_data):
    X_dummy = np.array(iris_data.loc[0, ['X1', 'X2', 'X3', 'X4']])
    X_inp = np.array(iris_data[['X1', 'X2', 'X3', 'X4']])
    og_preds = keras_model.predict(X_inp)

    f = os.path.join(os.path.dirname(__file__), "test_models/keras_example.onnx")
    onnx = convert_to_onnx(keras_model, "keras", X_dummy, output_path = f)
    sess = rt.InferenceSession(onnx.SerializeToString())
    label_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name
    print("Keras to ONNX serialized default output label:", label_name)
    print("Keras to ONNX serialized default input label:", input_name)
    onnx_preds = sess.run([label_name], {input_name: X_inp.astype(np.float32)})[0]

    np.testing.assert_array_almost_equal(og_preds, onnx_preds, decimal = 5)

@needs_tf
def test_tensorflow_conversion(ssd_data):
    import tensorflow as tf
    MODEL = "ssd_mobilenet_v1_coco_2018_01_28"

    # SavedModel convert
    convert_to_onnx(model = f"{model_path}/{MODEL}/saved_model", framework = "tensorflow",
                output_path = f"test_models/ssd_mobilenet_savedmodel.onnx", target_opset = 10)

    # Frozen graph convert
    convert_to_onnx(model = f"{model_path}/{MODEL}/frozen_inference_graph.pb", framework = "tensorflow",
                 output_path = f"test_models/ssd_mobilenet_frozengraph.onnx", tf_input_names = ["image_tensor:0"],
                 tf_output_names = ['num_detections:0', 'detection_boxes:0',
                                    'detection_scores:0','detection_classes:0'],
               target_opset = 10)

    # we want the outputs in this order
    outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

    import onnxruntime as rt
    sess = rt.InferenceSession("test_models/ssd_mobilenet_savedmodel.onnx")

    result = sess.run(outputs, {"image_tensor:0": ssd_data})
    onnx_num_detections, onnx_detection_boxes, onnx_detection_scores, onnx_detection_classes = result

    sess = rt.InferenceSession("test_models/ssd_mobilenet_frozengraph.onnx")

    result = sess.run(outputs, {"image_tensor:0": ssd_data})
    onnx_num_detections2, onnx_detection_boxes2, onnx_detection_scores2, onnx_detection_classes2 = result

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.saved_model.load(sess, ['serve'], f"{model_path}/{MODEL}/saved_model")
        num_detections, detection_boxes, detection_scores, detection_classes = sess.run(outputs, {"image_tensor:0": ssd_data})

    np.testing.assert_array_almost_equal(onnx_num_detections, num_detections, decimal=5)
    np.testing.assert_array_almost_equal(onnx_detection_boxes, detection_boxes, decimal=5)
    np.testing.assert_array_almost_equal(onnx_detection_scores, detection_scores, decimal=5)
    np.testing.assert_array_almost_equal(onnx_detection_classes, detection_classes, decimal=5)

    np.testing.assert_array_almost_equal(onnx_num_detections, onnx_num_detections2, decimal=5)
    np.testing.assert_array_almost_equal(onnx_detection_boxes, onnx_detection_boxes2, decimal=5)
    np.testing.assert_array_almost_equal(onnx_detection_scores, onnx_detection_scores2, decimal=5)
    np.testing.assert_array_almost_equal(onnx_detection_classes, onnx_detection_classes2, decimal=5)
