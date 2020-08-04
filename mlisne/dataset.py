from abc import ABC, abstractmethod
from pydantic.dataclasses import dataclass # Use pydantic for runtime type-checking
from dataclasses import InitVar
from pathlib import Path
from typing import Tuple, Dict, Union, Sequence, Optional
import warnings
import numpy as np
import pandas as pd
import os

class Config:
    arbitrary_types_allowed = True

class BaseEstimatorDataset(ABC):
    """Base MLisNE Dataset Class"""

    @abstractmethod
    def load_data(self) -> None:
        """Load raw data"""
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self) -> None:
        """Preprocess raw data"""
        raise NotImplementedError()

# TODO: TENSOR SUPPORT
@dataclass(config=Config)
class IVEstimatorDataset(BaseEstimatorDataset):
    """Class for loading and preprocessing data for IV method of treatment effect estimation.

    Parameters
    -----------
    data: InitVar[Union[str, np.ndarray, pd.DataFrame]]
        Data object or path to csv with columns assumed to be in order [Y, Z, D, X_d, X_c], unless other indices given
    Y: Union[int, np.ndarray, pd.Series, Sequence]
        Data object of outcome variable (float)
    Z: Union[int, np.ndarray, pd.Series, Sequence]
        Data object of binary ML assignment variable (int)
    D: Union[int, np.ndarray, pd.Series]
        Data object of binary treatment assignment variable (int)
    X_c: Union[np.ndarray, pd.DataFrame]
        Data object of continuous variables (float)
    X_d: Union[np.ndarray, pd.DataFrame]
        Data object of discrete variables (int)
    normalized: Boolean
        Indicator for if continuous data is normalized

    WARNING: if both C and X_d are not given, then all covariates are assumed to be continuous!
    """
    data: InitVar[Union[str, np.ndarray, pd.DataFrame]] = None
    Y: Union[int, np.ndarray, pd.Series, Sequence] = None
    Z: Union[int, np.ndarray, pd.Series, Sequence] = None
    D: Union[int, np.ndarray, pd.Series, Sequence] = None
    X_c: Union[np.ndarray, pd.Series, pd.DataFrame, Sequence] = None
    X_d: Union[np.ndarray, pd.Series, pd.DataFrame, Sequence] = None
    normalized: bool = False

    def __post_init__(self, data) -> None:
        """Initialize IV Estimator Dataset"""
        self.load_data(data, self.Y, self.Z, self.D, self.X_c, self.X_d)

    def load_data(self, data: InitVar[Union[str, np.ndarray, pd.DataFrame]] = None,
                        Y: Union[int, np.ndarray, pd.Series, Sequence] = None,
                        Z: Union[int, np.ndarray, pd.Series, Sequence] = None,
                        D: Union[int, np.ndarray, pd.Series, Sequence] = None,
                        X_c: Union[np.ndarray, pd.Series, pd.DataFrame, Sequence] = None,
                        X_d: Union[np.ndarray, pd.Series, pd.DataFrame, Sequence] = None) -> None:
        """
        If `data` is given, then the remaining arguments are expected to be indices of the relevant variables.
            Any missing indices will be inferred from the expected column order in `data`: [Y, Z, D, X_d, X_c].
            If neither X_c nor X_d are given, then the covariates are assumed to be all continuous.
        Otherwise, the remaining arguments are expected to be data objects for overwriting specific variables.
        """
        if data is not None:
            # TODO: Raise error if data does not have minimum number of columns
            if isinstance(data, str) or isinstance(data, os.PathLike):
                data = np.array(pd.read_csv(data))
            if isinstance(data, pd.DataFrame):
                data = np.array(data)
            infer = []
            indices_to_remove = []
            for key, val in locals().items():
                if key in ["self", "data", "infer"]:
                    continue
                if val is None:
                    infer.append(key)
                    continue
                if key in ['X_c', "X_d"]:
                    setattr(self,key, np.squeeze(data[:,val].astype(float)))
                else:
                    setattr(self, key, data[:,val])
                # Save indices to be removed
                if isinstance(val, Sequence):
                    indices_to_remove.extend(val)
                else:
                    indices_to_remove.append(val)
            data = np.delete(data, indices_to_remove, axis=1)
            if "X_c" in infer:
                if "X_d" in infer:
                    warnings.warn("Neither continuous nor discrete indices were explicitly given. We will assume all covariates in data are continuous.", stacklevel=2)
                    cts_start_ind = len(infer) - 2 # Account for X_d being left as None
                    infer.remove("X_d")
                else:
                    cts_start_ind = len(infer) - 1
                self.X_c = np.squeeze(data[:,cts_start_ind:].astype(float))
                infer.remove("X_c")
            elif "X_d" in infer:
                discrete_start_ind = len(infer) - 1
                self.X_d = np.squeeze(data[:,discrete_start_ind:].astype(float))
                infer.remove("X_d")
            for i in range(len(infer)):
                setattr(self, infer[i], data[:,i])
        # If data not given, then override data objects
        else:
            for key, val in locals().items():
                if key in ["self", "data", "infer"]:
                    continue
                if val is None or isinstance(val, int):
                    continue
                # TODO: Enforce data object type if `data` not passed!
                val = np.array(val)
                setattr(self, key, val)
    def preprocess(self) -> None:
        pass
