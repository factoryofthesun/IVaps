from pathlib import Path
from typing import Tuple, Dict, Union, Sequence, Optional
import onnxruntime as rt
import warnings
import numpy as np
import pandas as pd
import torch
from onnxmltools import convert_keras
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType, Int64TensorType

from mlisne.dataset import IVEstimatorDataset

def check_is_fitted(estimator):
    return

def convert_to_onnx(model, dummy_input, path: str, framework: str, input_type: int = 1, **kwargs) -> None:
    """Convenience wrapper to quickly convert to ONNX model with expected arguments"""
    if framework == "sklearn":
        if input_type == 1:
            tensortype = _guess_numpy_type(dummy_input.dtype)
            initial_type = [('input', tensortype([None, len(dummy_input)]))]
        elif input_type == 2:
            c_tensortype = _guess_numpy_type(dummy_input[0].dtype)
            d_tensortype = _guess_numpy_type(dummy_input[1].dtype)
            initial_type = [('c_inputs', c_tensortype([None, len(dummy_input[0])])),
                            ('d_inputs', d_tensortype([None, len(dummy_input[1])]))]
        else:
            raise ValueError("input_type must be either 1 or 2")
        onx = convert_sklearn(model, initial_types=initial_type, **kwargs)
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())
        return
    if framework == "pytorch":
        if input_type == 1:
            torch.onnx.export(model, dummy_input, path, input_names=['input'], output_names=['output_probability'],
                              dynamic_axes={'inputs':{0:'N'},'output_probability':{0, 'N'}})
        elif input_type == 2:
            torch.onnx.export(model, dummy_input, path, input_names=['c_inputs', 'd_inputs'], output_names=['output_probability'],
                              dynamic_axes={'c_inputs':{0:'N'}, 'd_inputs':{0:'N'}, 'output_probability':{0, 'N'}})
        else:
            raise ValueError("input_type must be either 1 or 2")
    else:
        print(f"{framework} conversion not yet implemented for this function."
               "Please see https://github.com/onnx/onnxmltools for more conversion functions.")
def _guess_numpy_type(data_type):
    if data_type == np.float32:
        return FloatTensorType
    if data_type == np.float64:
        return DoubleTensorType
    if data_type == np.str:
        return StringTensorType
    if data_type in (np.int64, np.uint64):
        return Int64TensorType
    if data_type in (np.int32, np.uint32):
        return Int32TensorType
    if data_type == np.bool:
        return BooleanTensorType
    raise NotImplementedError(
        "Unsupported data_type '{}'. You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "".format(data_type))
