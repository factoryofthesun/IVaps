from abc import ABC, abstractmethod
from pydantic.dataclasses import dataclass # Use pydantic for runtime type-checking
from dataclasses import InitVar
from pathlib import Path
from typing import Tuple, Dict, Union, Sequence, Optional
import warnings
import numpy as np
import pandas as pd

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

@dataclass
class IVEstimatorDataset(BaseEstimatorDataset):
    """Class for loading and preprocessing data for IV method of treatment effect estimation.

    Parameters
    -----------
    data: InitVar[Union[str, np.ndarray, pd.DataFrame]]
        Data object or path to csv with columns assumed to be in order [Y, Z, D, X_d, X_c], unless C given
    C: InitVar[Sequence]
        List of column indices of continuous variables in X
    Y: Union[np.ndarray, pd.Series]
        Data object of outcome variable
    Z: Union[np.ndarray, pd.Series]
        Data object of binary ML assignment variable
    D: Union[np.ndarray, pd.Series]
        Data object of binary treatment assignment variable
    X_c: Union[np.ndarray, pd.DataFrame]
        Data object of continuous variables
    X_d: Union[np.ndarray, pd.DataFrame]
        Data object of discrete variables

    WARNING: if both C and X_d are not given, then all covariates are assumed to be continuous!
    """
    data: InitVar[Union[str, np.ndarray, pd.DataFrame]] = None
    C: InitVar[Sequence] = None
    Y: Union[np.ndarray, pd.Series, Sequence] = None
    Z: Union[np.ndarray, pd.Series, Sequence] = None
    D: Union[np.ndarray, pd.Series, Sequence] = None
    X_c: Union[np.ndarray, pd.Series, Sequence] = None
    X_d: Union[np.ndarray, pd.Series, Sequence] = None

    def __post_init__(self, data, C) -> None:
        """Initialize IV Estimator Dataset"""
        self.load_data(data, C)

    def load_data(self, data: Union[str, np.ndarray, pd.DataFrame] = None,
                        C: Sequence = None,
                        Y: Union[np.ndarray, pd.Series, Sequence] = None,
                        Z: Union[np.ndarray, pd.Series, Sequence] = None,
                        D: Union[np.ndarray, pd.Series, Sequence] = None,
                        X_c: Union[np.ndarray, pd.DataFrame, pd.Series, Sequence] = None,
                        X_d: Union[np.ndarray, pd.DataFrame, pd.Series, Sequence] = None) -> None:
        """
        Overwrite object data with an optional amount of IV Regression inputs.
        All attributes are cast to numpy arrays. If data is given, then all values are overwritten with the values in data.
        If C is not given, then the positions of the continuous variables are inferred to be after all the other variables.
        If input is None, then attribute is kept as is but cast to numpy array or list.
        """
        filled_vars = []
        for key, val in locals().items():
            if key in ["self", "data", "C"]:
                continue
            if val is not None:
                filled_vars.append(key)
                setattr(self, key, np.array(val))
            else:
                val = self.__dict__[key]
                setattr(self, key, np.array(val))

        # Cast data to numpy array for loading purposes
        if isinstance(data, str):
            data = np.array(pd.read_csv(data))
        if isinstance(data, pd.DataFrame):
            data = np.array(data)
        if data is not None:
            # Interpret columns based off of filled inputs
            # If indices of continuous variables are given, then overwrite the remainder of the unfilled attributes
            # in the asssumed order.
            unfilled_vars = [var for var in self.__dict__.keys() if var not in filled_vars]
            cts_start_ind = 0
            if C is not None:
                self.X_c = data[:,C]
                # Set attributes for whatever was not explicitly overwritten
                data = np.delete(data, C, axis=1)
                if len(unfilled_vars) > 0:
                    if "X_d" in unfilled_vars:
                        discrete_start_ind = len(unfilled_vars) - 1
                        self.X_d = data[:,discrete_start_ind:]
                        unfilled_vars.remove("X_d")
                    for i in range(len(unfilled_vars)):
                        setattr(self, unfilled_vars[i], data[:,i])
            # ==== If indices are not given, then infer from the assumed order ====
            # Starting index of continous vars is index after final X_d column
            elif "X_d" in filled_vars:
                if self.X_d.ndim == 1:
                    cts_start_ind += 1
                else:
                    cts_start_ind += self.X_d.shape[1]
            else:  # If both X_d and C are not given, then all the covariates are assumed to be continuous
                warnings.warn("Neither continuous indices nor discrete variables were explicitly given. We will assume all covariates in data are continuous.")
            cts_start_ind += len(unfilled_vars)-1
            self.X_c = data[:,cts_start_ind:]
            for i in range(len(unfilled_vars)):
                setattr(self, unfilled_vars[i], data[:,i])
