"""Helper functions"""
from pathlib import Path
from typing import Tuple, Dict, Union, Sequence, Optional
import onnxruntime as rt
import warnings
import numpy as np
import pandas as pd
import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto
from onnxmltools.convert.common.data_types import FloatTensorType, DoubleTensorType, Int64TensorType, Int32TensorType, StringTensorType, BooleanTensorType
from numba import jit, njit

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
    np.ndarray
        Outputs of ONNX inference or post-inference function

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

    # Remove unnecessary dims
    ml_out = np.squeeze(ml_out)

    return ml_out

def convert_to_onnx(model, framework: str, dummy_input1 = None, dummy_input2 = None, output_path: str = None,
                    input_names: Tuple[str, str] = ("c_inputs", "d_inputs"), output_names: Sequence = None,
                    tf_input_names: Sequence = None, tf_output_names: Sequence = None,
                    target_opset: int = None, **kwargs):
    """Convenience function to quickly convert and save ONNX model with expected input/output settings

    Parameters
    -----------
    model: object
        fitted model object (or path to saved model in Tensorflow case)
    framework: str
        Reference string for one of the implemented frameworks
    dummy_input1: list-like, default: None
        Dummy input for first model input used for type inference and passed into downstream conversion functions
    dummy_input2: list-like, default: None
        Dummy input for second model input (if applicable) used for type inference and passed into downstream conversion functions
    output_path: str, default: None
        path to save ONNX model
    input_names: Tuple[str, str], default: ("c_inputs", "d_inputs")
        input names to assign ONNX model
    output_names: list-like, default: None
        output names for later ONNX inference; if None defaults to naming the outputs sequentially "output_1", "output_2", etc...
    tf_input_names: list-like, default: None
        Input names for Tensorflow graph. Only required when converting from Tensorflow using a frozen graph or checkpoints.
    tf_output_names: list-like, default: None
        Output names for Tensorflow graph. Only required when converting from Tensorflow using a frozen graph or checkpoints.
    **kwargs: keyword arguments to be passed into mltools conversion function

    Returns
    -----------
    Object
        Converted ONNX model or boolean flag indicating successful conversion, depending on specific framework.

    """

    # Adjust dummy input(s) if incorrect dimension
    if dummy_input1 is not None:
        if framework != "pytorch":
            dummy_input1 = np.array(dummy_input1)
            if dummy_input1.ndim > 1:
                dummy_input1 = dummy_input1[0].flatten()
        else:
            if dummy_input1.ndim == 1:
                dummy_input1 = dummy_input1[:, np.newaxis]

    if dummy_input2 is not None:
        if framework != "pytorch":
            dummy_input2 = np.array(dummy_input2)
            if dummy_input2.ndim > 1:
                dummy_input2 = dummy_input2[0].flatten()
        else:
            if dummy_input2.ndim == 1:
                dummy_input2 = dummy_input2[:, np.newaxis]

    if framework in ["sklearn",
                     "lightgbm",
                     "xgboost",
                     "catboost",
                     "coreml",
                     "libsvm",
                     "sparkml",
                     "keras",]:
        if dummy_input1 is None:
            raise ValueError(f"Conversion from {framework} model requires a dummy input.")
        elif dummy_input2 is None:
            tensortype = _guess_numpy_type(dummy_input1.dtype)
            initial_type = [(input_names[0], tensortype([None, len(dummy_input1)]))]
        else:
            tensortype_1 = _guess_numpy_type(dummy_input1.dtype)
            tensortype_2 = _guess_numpy_type(dummy_input2.dtype)
            initial_type = [(input_names[0], tensortype_1([None, len(dummy_input1)])),
                                (input_names[1], tensortype_2([None, len(dummy_input2)]))]

        if framework == "sklearn":
            from onnxmltools import convert_sklearn
            onx = convert_sklearn(model, initial_types=initial_type, target_opset = target_opset, **kwargs)
        if framework == "keras":
            from onnxmltools import convert_keras
            onx = convert_keras(model, initial_types=initial_type, target_opset = target_opset, **kwargs)
        if framework == "lightgbm":
            from onnxmltools import convert_lightgbm
            onx = convert_lightgbm(model, initial_types=initial_type, target_opset = target_opset, **kwargs)
        if framework == "xgboost":
            from onnxmltools import convert_xgboost
            onx = convert_xgboost(model, initial_types=initial_type, target_opset = target_opset, **kwargs)
        if framework == "catboost":
            from onnxmltools import convert_catboost
            onx = convert_catboost(model, initial_types=initial_type, target_opset = target_opset, **kwargs)
        if framework == "coreml":
            from onnxmltools import convert_coreml
            onx = convert_coreml(model, initial_types=initial_type, target_opset = target_opset, **kwargs)
        if framework == "libsvm":
            from onnxmltools import convert_libsvm
            onx = convert_libsvm(model, initial_types=initial_type, target_opset = target_opset, **kwargs)
        if framework == "sparkml":
            from onnxmltools import sparkml
            onx = convert_sparkml(model, initial_types=initial_type, target_opset = target_opset, **kwargs)

        # TODO: Rename outputs -- BELOW DOESNT WORK
        # if output_names is None:
        #     for i in range(len(onx.graph.output)):
        #         onx.graph.output[i].name = f"output_{i}"
        # else:
        #     for i in range(len(onx.graph.output)):
        #         onx.graph.output[i].name = f"{output_names[i]}"

        if output_path is not None:
            with open(output_path, "wb") as f:
                f.write(onx.SerializeToString())
        return onx

    # Models that don't use intial_types
    if framework in ["cntk",]:
        if framework == "cntk":
            import cntk
            if output_path is None:
                raise ValueError(f"Conversion from {framework} requires output_path.")
            model.save(output_path, format=cntk.ModelFormat.ONNX)

    if framework == "pytorch":
        import torch
        from torch.onnx import export

        if output_path is None:
            raise ValueError(f"Conversion from {framework} requires output_path.")
        if dummy_input1 is None:
            raise ValueError(f"Conversion from {framework} model requires a dummy input.")
        if dummy_input2 is None:
            d_axes = {input_names[0]:{0:'N'}}
            if output_names is None:
                output_names = ["output_0"]
                d_axes.update({"output_0": {0:'N'}})
            else:
                d_axes.update({key: {0:'N'} for key in output_names})
            # Convert to tensor if necessary
            if not isinstance(dummy_input1, torch.Tensor):
                dummy_input1 = torch.tensor(dummy_input1)
            export(model, dummy_input1, output_path, input_names=[input_names[0]], output_names=output_names,
                              dynamic_axes=d_axes, **kwargs)
        else:
            if not isinstance(dummy_input1, torch.Tensor):
                dummy_input1 = torch.tensor(dummy_input1)
            if not isinstance(dummy_input2, torch.Tensor):
                dummy_input2 = torch.tensor(dummy_input2)
            dummy_input = (dummy_input1, dummy_input2)
            d_axes = {input_names[0]:{0:'N'}, input_names[1]:{0:'N'}}
            if output_names is None:
                output_names = ["output_0"]
                d_axes.update({"output_0": {0:'N'}})
            else:
                d_axes.update({key: {0:'N'} for key in output_names})
            export(model, dummy_input, output_path, input_names=list(input_names), output_names=output_names,
                              dynamic_axes=d_axes, target_opset = target_opset, **kwargs)
        return True

    if framework == "tensorflow":
        from onnxmltools import convert_tensorflow
        from mlisne.utils import get_extension
        import subprocess
        import tensorflow as tf

        if output_path is None:
            raise ValueError(f"Conversion from {framework} requires output_path.")
        if not isinstance(model, str):
            raise ValueError(f"Conversion from {framework} requires `model` to be a str path.")
        if target_opset is None:
            from onnxconverter_common.onnx_ex import get_maximum_opset_supported
            target_opset = get_maximum_opset_supported()

        target_opset = str(target_opset)

        # Run tf2onnx conversion
        # TODO: PIPE SUBPROCESS OUTPUT TO STDOUT
        if get_extension(model) == "pb":
            if not tf_input_names or not tf_output_names:
                raise ValueError(
                    "Please provide --model_inputs_names and --model_outputs_names to convert Tensorflow graphdef models.")
            # Convert input vars to string
            tf_input_names = ",".join(tf_input_names)
            tf_output_names = ",".join(tf_output_names)
            call = ["python", "-m", "tf2onnx.convert", "--input", model, "--output", output_path, "--inputs",
                    tf_input_names, "--outputs", tf_output_names, "--opset", target_opset,
                    "--fold_const", "--target", "rs6"]
            subprocess.check_call(call)
        elif get_extension(model) == "meta":
            if not tf_input_names or not tf_output_names:
                raise ValueError(
                    "Please provide --model_inputs_names and --model_outputs_names to convert Tensorflow graphdef models.")
            # Convert input vars to string
            tf_input_names = ",".join(tf_input_names)
            tf_output_names = ",".join(tf_output_names)
            call = ["python", "-m", "tf2onnx.convert", "--checkpoint", model, "--output", output_path, "--inputs",
                    tf_input_names, "--outputs", tf_output_names, "--opset", target_opset,
                    "--fold_const", "--target", "rs6"]
            subprocess.check_call(call)
        else:
            call = ["python", "-m", "tf2onnx.convert", "--saved-model", model, "--output", output_path,
                    "--opset", target_opset, "--fold_const", "--target", "rs6"]
            subprocess.check_call(call)
        return True

    else:
        print(f"{framework} conversion not yet implemented for this function."
               "Please see https://github.com/onnx/onnxmltools for more conversion functions.")
        return False

# Wraps check_model function in onnx_converter
def check_conversion(model_path: str, onnx_model_path: str, framework: str, test_input_path: str = None,
                     tf_input_names: Sequence = None, tf_output_names: Sequence = None,
                     log_path: str = None):
    """Check successful conversion of ONNX model

    Parameters
    -----------
    model_path: str
        Path to original saved model
    onnx_model_path: str
        Path to converted ONNX model
    framework: str
        Reference string for one of the implemented frameworks
    test_input_path: str, default: None
        Path to folder with saved .pb test inputs
    tf_input_names: Sequence, default: None
        Names of inputs for Tensorflow model, if applicable
    tf_output_names: Sequence, default: None
        Names of outputs for Tensorflow model, if applicable
    log_path: str, default: None
        Path to save test results

    Returns
    -----------
    bool
        True if model passses all checks

    """
    from mlisne.utils import check_model, generate_inputs

    output_template = {
    "output_onnx_path": onnx_model_path,  # The output path where the converted .onnx file is stored.
    "correctness_verified": "",  # SUCCEED, NOT SUPPORTED, SKIPPED
    "input_folder": "",
    "error_message": ""
    }

    # Generate random inputs for the model if input files are not provided
    try:
        # Will search for .pb files in `test_input_path` if not None, otherwise checks `model_path`
        # and copies the files over to `test_data_set_0` in the onnx model directory if not already created
        inputs_path = generate_inputs(model_path, test_input_path, onnx_model_path)
        output_template["input_folder"] = inputs_path
    except Exception as e:
        output_template["error_message"] = str(e)
        output_template["correctness_verified"] = "SKIPPED"
        print("\n-------------\nMODEL CONVERSION SUMMARY\n")
        print(output_template)
        if log_path is not None:
            print(f"Writing log output to {log_path}...")
            with open(log_path, "w") as f:
                json.dump(output_template, f, indent=4)
        raise e

    print("\n-------------\nMODEL CORRECTNESS VERIFICATION\n")
    # Test correctness: check_model can be called with arbitrary inputs without output as well
    # Get saved ONNX model input names
    verify_status = check_model(model_path, onnx_model_path, inputs_path, framework,
                                tf_input_names, tf_output_names)
    output_template["correctness_verified"] = verify_status

    print("\n-------------\nMODEL CONVERSION SUMMARY\n")
    print(output_template)
    if log_path is not None:
        print(f"Writing log output to {log_path}...")
        with open(output_json_path, "w") as f:
            json.dump(output_template, f, indent=4)

    return True

def convert_data_to_pb(pickle_path: str, output_folder: str ="test_data_set_0", is_input=True):
    """ Convert pickle test data file to ONNX .pb files.

    Parameters
    -----------
    pickle_path: str
        The path to your pickle file. The pickle file should contain a dictionary with the following format:
        \\{
        input_name_1: test_data_1,
        input_name_2: test_data_2,
        ...
        \\}
    output_folder: str, default: "test_data_set_0"
        The folder to store .pb files. The folder should be empty and its name starts with test_data_*.

    """
    import pickle, os

    extension = pickle_path.split(".")[1]
    if extension == "pb":
        print("Test Data already in .pb format. ")
        return
    try:
        test_data_dict = pickle.load(open(pickle_path, "rb"))
    except:
        raise ValueError("Cannot load test data with pickle. ")
    # Type check for the pickle file. Expect a dictionary with input names as keys
    # and data as values.
    if type(test_data_dict) is not dict:
        raise ValueError("Data type error. Expect a dictionary with input names as keys and data as values.")

    # Create a test_data_set folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_prefix = "input_"
    if is_input == False:
        file_prefix = "output_"
    idx = 0
    for name, data in test_data_dict.items():
        tensor = numpy_helper.from_array(data)
        tensor.name = name
        pb_file_name = file_prefix + f"{idx}.pb"
        pb_file_location = os.path.join(output_folder, pb_file_name)
        with open(pb_file_location, 'wb') as f:
            f.write(tensor.SerializeToString())
            print("Successfully stored input {} in {}".format(name, pb_file_location))
        idx += 1

@jit(parallel=True)
def standardize(X):
    """ Standardize 2D array of variables """
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    X = (X - mu)/sigma
    return (X, mu, sigma)

@jit(parallel = True)
def cumMean(X, S, is_list = False):
    """ Return mean of every S rows """
    if is_list == True:
        nobs = int(X[0].shape[0]/S)
        ret = np.empty((nobs, 0))
        for x_tmp in X:
            i = 0
            ret_tmp = []
            while (i+1)*S <= x_tmp.shape[0]:
                ret_tmp.append(np.mean(x_tmp[(i*S):(i+1)*S]))
                i += 1
            ret_tmp = np.array(ret_tmp)
            ret = np.column_stack((ret, ret_tmp))
    else:
        i = 0
        ret = []
        while (i+1)*S <= X.shape[0]:
            ret.append(np.mean(X[(i*S):(i+1)*S]))
            i += 1
        ret = np.array(ret)
    return ret

def _olive_convert(model_name: str, framework: str, test_data_path: str = None, convert_from_pickle: bool = False, input_pickle: str = None,
                  output_pickle: str = None, output_folder: str = None,
                  model_path: str = "./", convert_directory: str = "./", convert_name: str = None, update_sdk: bool = True, **kwargs):
    import os
    import wget
    import subprocess

    url = "https://raw.githubusercontent.com/microsoft/OLive/master/utils/"
    sdk_files = ["onnxpipeline.py", "convert_test_data.py", "config.py"]
    sdk_dir = "./python_sdk"
    if not os.path.exists(sdk_dir):
        os.makedirs(sdk_dir)

    if update_sdk == True:
        for filename in sdk_files:
            target_file = os.path.join(sdk_dir, filename)
            if not os.path.exists(target_file) or update_sdk == True:
                print("Downloading OLive Python SDK files...")
                wget.download(url + filename, target_file)
                print("Downloaded", filename)

    # Pull latest onnx-converter image from mcr
    print("Pulling latest onnx-converter image...")
    subprocess.run(["docker", "pull", "mcr.microsoft.com/onnxruntime/onnx-converter"])

    # Convert test data if toggled -- output path will be same directoy as converted model
    if convert_from_pickle  == True:
        pass

    # Initiate conversion pipeline in convert directory
    sys.path.append("./python_sdk")
    import onnxpipeline

    pipeline = onnxpipeline.Pipeline(model_path, convert_directory = convert_directory, convert_name = convert_name)

    # Different frameworks require different inputs
    model = pipeline.convert_model(model = model_name, model_type = framework)

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
