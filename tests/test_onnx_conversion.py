# Test ONNX conversion
import sys
import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from sklearn.datasets import load_iris
import onnxruntime as rt
from pathlib import Path

from mlisne.helpers import convert_to_onnx


model_path = str(Path(__file__).resolve().parents[1] / "examples" / "models")
data_path = str(Path(__file__).resolve().parents[1] / "examples" / "data")

@pytest.fixture
def 
