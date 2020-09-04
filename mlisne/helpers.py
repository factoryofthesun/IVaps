"""Helper functions"""

from pathlib import Path
from typing import Tuple, Dict, Union, Sequence, Optional
import onnxruntime as rt
import warnings
import numpy as np
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType, DoubleTensorType, Int64TensorType

def check_is_fitted(estimator):
    return

def run_onnx_session(inputs: Sequence[np.ndarray], sess: rt.InferenceSession, input_names: Sequence[str],
                     label_names: Sequence[str] = None, fcn = None, **kwargs):
    """Convenience function to execute ONNX inference with an optional post-inference function

    Parameters
    -----------
    inputs: Sequence of array-likes
        ONNX inference inputs
    sess: onnxruntime InferenceSession
    input_names: Sequence of strings
        Input names to assign to inputs
    label_names: Sequence of strings, default: all outputs
        Specific outputs to return from inference
    fcn: Object, default: None
        Vectorized function to pass inference outputs through
    **kwargs: additional arguments to pass into fcn

    Returns
    -----------
    Outputs of ONNX inference or fcn

    """

    feed_dict = dict(zip(input_names, inputs))
    ml_out = sess.run(label_names, feed_dict)

    # All outputs are wrapped in a list -- if single label then send to 1d list
    if len(label_names) == 1:
        ml_out = ml_out[0]

    # Account for case in which output probabilities are in dictionary of class labels
    if isinstance(ml_out[0], Dict):
        ml_out = np.array([d[1] for d in ml_out])

    if fcn is not None:
        ml_out = fcn(ml_out, **kwargs)

    return ml_out

def convert_to_onnx(model, dummy_input, path: str, framework: str, input_type: int = 1,
                    input_names: Tuple[str, str] = ("c_inputs", "d_inputs"),
                    output_names: Sequence = None, **kwargs) -> bool:
    """Convenience function to quickly convert and save ONNX model with expected input/output settings

    Parameters
    -----------
    model: object
        fitted model object
    dummy_input: 1D input array or 2-tuple of continuous and discrete input arrays
        Used for type inference and passed into downstream conversion functions
    path: str
        path to save ONNX model
    framework: str
        Reference string for one of the implemented frameworks
    input_type: int, default: 1
        1 if single array input, 2 if model takes continuous and discrete values separately
    input_names: Tuple[str, str], default: ("c_inputs", "d_inputs")
        input names to assign ONNX model
    output_names: list-like, default: None
        output names for later ONNX inference; if None defaults to naming the outputs sequentially "output_1", "output_2", etc...
    **kwargs: keyword arguments to be passed into mltools conversion function

    Returns
    -----------
    bool
        Flag indicating successful conversion

    """

    if framework in ["sklearn",
                     "lightgbm",
                     "xgboost",
                     "catboost",
                     "coreml",
                     "libsvm",
                     "sparkml",]:
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

        if framework == "sklearn":
            from onnxmltools import convert_sklearn
            onx = convert_sklearn(model, initial_types=initial_type, **kwargs)
        if framework == "lightgbm":
            from onnxmltools import convert_lightgbm
            onx = convert_lightgbm(model, initial_types=initial_type, **kwargs)
        if framework == "xgboost":
            from onnxmltools import convert_xgboost
            onx = convert_xgboost(model, initial_types=initial_type, **kwargs)
        if framework == "catboost":
            from onnxmltools import convert_catboost
            onx = convert_catboost(model, initial_types=initial_type, **kwargs)
        if framework == "coreml":
            from onnxmltools import convert_coreml
            onx = convert_coreml(model, initial_types=initial_type, **kwargs)
        if framework == "libsvm":
            from onnxmltools import convert_libsvm
            onx = convert_libsvm(model, initial_types=initial_type, **kwargs)
        if framework == "sparkml":
            from onnxmltools import sparkml
            onx = convert_sparkml(model, initial_types=initial_type, **kwargs)

        # TODO: Rename outputs -- BELOW DOESNT WORK
        # if output_names is None:
        #     for i in range(len(onx.graph.output)):
        #         onx.graph.output[i].name = f"output_{i}"
        # else:
        #     for i in range(len(onx.graph.output)):
        #         onx.graph.output[i].name = f"{output_names[i]}"

        with open(path, "wb") as f:
            f.write(onx.SerializeToString())
        return True

    if framework == "pytorch":
        from torch.onnx import export
        #TODO: How to check # of outputs?
        if input_type == 1:
            d_axes = {input_names[0]:{0:'N'}}
            if output_names is None:
                d_axes.update({"output_0": {0:'N'}})
            else:
                d_axes.update({key: {0:'N'} for key in output_names})
            export(model, dummy_input, path, input_names=[input_names[0]], output_names=["output_0"],
                              dynamic_axes=d_axes, **kwargs)
        elif input_type == 2:
            d_axes = {input_names[0]:{0:'N'}, input_names[1]:{0:'N'}}
            if output_names is None:
                d_axes.update({"output_0": {0:'N'}})
            else:
                d_axes.update({key: {0:'N'} for key in output_names})
            export(model, dummy_input, path, input_names=list(input_names), output_names=["output_0"],
                              dynamic_axes=d_axes, **kwargs)
        else:
            raise ValueError("input_type must be either 1 or 2")
        return True

    if framework == "keras":
        from onnxmltools import convert_keras
        onx = convert_keras(model, **kwargs)
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())
        return True

    else:
        print(f"{framework} conversion not yet implemented for this function."
               "Please see https://github.com/onnx/onnxmltools for more conversion functions.")
        return False

def _guess_numpy_type(data_type):
    """Guess the ONNX tensortype from the given numpy dtype"""

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

class Config:
    """Configuration class for Pydantic Dataclass"""
    arbitrary_types_allowed = True
