from pathlib import Path
from typing import Tuple, Dict, Union, Sequence, Optional
import onnxruntime as rt
import warnings
import numpy as np
import pandas as pd
from onnxmltools import convert_keras, convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType, DoubleTensorType, Int64TensorType

from mlisne.dataset import IVEstimatorDataset

def check_is_fitted(estimator):
    return

def convert_to_onnx(model, dummy_input, path: str, framework: str, input_type: int = 1,
                    input_names: Tuple[str, str] = ("input",), **kwargs) -> bool:
    """Convenience wrapper to quickly convert and save ONNX model with expected input/output settings

    Arguments
    ---------
    model: fitted model object
    dummy_input: 1D input array or 2-tuple of continuous and discrete input arrays; for type inference
    path: string path to save ONNX model
    framework: one of the currently implemented frameworks {"sklearn", "pytorch"}
    input_type: 1 if single array input, 2 if model takes continuous and discrete values separately
    input_names: tuple of input names for later ONNX inference
    **kwargs: keyword arguments to be passed into mltools conversion function

    Returns: Boolean flag indicating successful conversion
    """
    if framework == "sklearn":
        if input_type == 1:
            tensortype = _guess_numpy_type(dummy_input.dtype)
            initial_type = [(input_names[0], tensortype([None, len(dummy_input)]))]
        elif input_type == 2:
            tensortype_1 = _guess_numpy_type(dummy_input[0].dtype)
            tensortype_2 = _guess_numpy_type(dummy_input[1].dtype)

            initial_type = [(input_names[0], tensortype_1([None, len(dummy_input[0])])),
                            (input_names[1], tensortype_2([None, len(dummy_input[1])]))]
        else:
            raise ValueError("input_type must be either 1 or 2")
        onx = convert_sklearn(model, initial_types=initial_type, **kwargs)
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())
        return True
    if framework == "pytorch":
        from torch.onnx import export
        if input_type == 1:
            export(model, dummy_input, path, input_names=[input_names[0]], output_names=['output_probability'],
                              dynamic_axes={'input':{0:'N'},'output_probability':{0:'N'}}, **kwargs)
        elif input_type == 2:
            export(model, dummy_input, path, input_names=list(input_names), output_names=['output_probability'],
                              dynamic_axes={'c_inputs':{0:'N'}, 'd_inputs':{0:'N'}, 'output_probability':{0:'N'}}, **kwargs)
        else:
            raise ValueError("input_type must be either 1 or 2")
        return True
    else:
        print(f"{framework} conversion not yet implemented for this function."
               "Please see https://github.com/onnx/onnxmltools for more conversion functions.")
        return False

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
