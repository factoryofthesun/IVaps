# Machine Learning is Natural Experiment
**[Overview](#overview)** | **[Installation](#installation)** | **[Usage](#usage)** | **[Acknowledgements](#acknowledgements)**
<details>
<summary><strong>Table of Contents</strong></summary>

- [Overview](#overview)
  - [ML as a Data Production Service](#ml-as-a-data-production)
  - [Framework](#framework)
  - [MLisNE Package](#mlisne-package)
    - [Supported ML Frameworks](#supported-ml-frameworks)
- [Installation](#installation)
  - [Requirements](#requirements)
- [Usage](#usage)
  - [Data Loading](#data-loading)
  - [QPS Estimation](#qps-estimation)
  - [IV Estimation](#iv-estimation)
  - [Model Conversion](#model-conversion)
  - [Futher Examples](#further-examples)
- [Versioning](#versioning)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
  - [References](#references)
  - [Related Projects](#related-projects)
</details>

# Overview 

## ML as a Data Production Service

Today’s society increasingly resorts to machine learning (“AI”) and other algorithms for decisionmaking and resource allocation. For example, judges make legal judgements using predictions from supervised machine learning (descriptive regression). Supervised learning is also used by governments to detect potential criminals and terrorists, and financial companies (such as banks and insurance companies) to screen potential customers. Tech companies like Facebook, Microsoft, and Netflix allocate digital content by reinforcement learning and bandit algorithms. Uber and other ride sharing services adjust prices using their surge pricing algorithms to take into account local demand and supply information. Retailers and e-commerce platforms like Amazon engage in algorithmic pricing. Similar algorithms are invading into more and more high-stakes treatment assignment, such as education, health, and military.

All of the above, seemingly diverse examples share a common trait: An algorithm makes decisions based only on observable input variables the data-generating algorithm uses. Conditional on the observable variables, therefore, algorithmic treatment decisions are (quasi-)randomly assigned. This property makes algorithm-based treatment decisions **an instrumental variable we can use for measuring the causal effect of the final treatment assignment**. The algorithm-based instrument may produce regression-discontinuity-style local variation (e.g. machine judges), stratified randomization (e.g. several bandit and reinforcement leaning algorithms), or mixes of the two.

## Framework
<img src="/images/ml_natural_experiment_diagram.PNG" width="550" height="250"/>
<img src="/images/framework_1.PNG" width="500" height="300"/>
<img src="/images/framework_2.PNG" width="500" height="300"/>
<img src="/images/framework_3.PNG" width="500" height="300"/>
<img src="/images/qps_chart.PNG" width="350" height="250"/>
<img src="/images/qps_estimation.PNG" width="500" height="300"/>

## MLisNE Package

The mlisne package is an implementation of the treatment effect estimation method described above. This package provides a simple-to-use pipeline for data preprocessing, QPS estimation, and treatment effect estimation that is ML framework-agnostic. 

### Supported ML Frameworks

The QPS estimation function `estimate_qps` only accepts models in the ONNX framework in order to maintain the framework-agnostic implementation. The module provides a `convert_to_onnx` function that currently supports conversion from the following frameworks:

- [Sklearn](https://github.com/onnx/sklearn-onnx/)
- [Pytorch](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [LightGBM](https://github.com/microsoft/LightGBM)
- [XGBoost (experimental)](https://github.com/dmlc/xgboost)
- [CatBoost (experimental)](https://github.com/catboost/catboost)
- [CoreML (experimental)](https://github.com/apple/coremltools)
- [LibSVM (experimental)](https://github.com/cjlin1/libsvm)
- [SparkML (experimental)](https://spark.apache.org/mllib/)
- [Keras (experimental)](https://keras.io/)

For conversion functions for other frameworks, please refer to the [onnxmltools repository](https://github.com/onnx/onnxmltools).
Please note that `convert_to_onnx` requires that the relevant framework packages are installed.

# Installation 
This package is still in its development phase, but you can compile the package from source
```bash
git clone https://github.com/factoryofthesun/mlisne
cd mlisne
pip install .
```
To install in development mode
```bash
git clone https://github.com/factoryofthesun/mlisne
cd mlisne
pip install -e ./
```
# Requirements

# Usage
Below is a general example of how to use the modules in this package to estimate LATE, given a converted ONNX model and historical treatment data.
```python
import pandas as pd
import numpy as np
import onnxruntime as rt
from mlisne.dataset import IVEstimatorDataset
from mlisne.qps import estimate_qps
from mlisne.estimator import TreatmentIVEstimator

# Read and load data
data = pd.read_csv("path_to_your_historical_data.csv")
iv_data = IVEstimatorDataset(data)

# Estimate QPS with 100 draws and ball radius 0.8
qps = estimate_qps(iv_data, S=100, delta=0.8, ML_onnx="path_to_onnx_model.onnx")

# Fit 2SLS Estimation model
estimator = TreatmentIVEstimator()
estimator.fit(iv_data, qps)

# Prints estimation results
print(estimator) 
estimator.firststage_summary() # Print summary of first stage results

estimator.coef # Array of second-stage estimated coefficients
estimator.varcov # Variance covariance matrix
```
In general the pipeline has three main steps: loading, QPS estimation, and IV estimation. 

## Data Loading
The IVEstimatorDataset class is the main data loader for the rest of the pipeline. It splits the data into individual arrays of the outcome `Y`, treatment assignment `D`, algorithmic recommendation `Z`, continuous inputs `X_c`, and discrete inputs `X_d`. The module can be initialized by passing in a pandas dataframe or numpy array with associated indices, or the variables can be individually assigned. 

```python
import pandas as pd
import numpy as np
from mlisne.dataset import IVEstimatorDataset

data = pd.read_csv("path_to_your_historical_data.csv")

# We can initialize the dataset by passing in the entire dataframe with indicator indices
iv_data = IVEstimatorDataset(data, Y=0, Z=1, D=2, X_c=range(3,6), X_d=range(6,9))

# We can also load the data post-initialization
iv_data = IVEstimatorDataset()
iv_data.load_data(Y=data['Y'], Z=data['Z'], D=data['D'], X_c=data.iloc[:,3:6], X_d=data.iloc[:,6:9])

# Indices that are not passed will be inferred from the remaining columns
iv_data = IVEstimatorDataset(data, Y=0, Z=1, D=2, X_c=range(3,6)) 
iv_data.X_d # data.iloc[:,6:9]

# If neither X_c nor X_d indices are given, then the input data is assumed to be all continuous
iv_data = IVEstimatorDataset(data, Y=0, Z=1, D=2) 
# "UserWarning: Neither continuous nor discrete indices were explicitly given. We will assume all covariates in data are continuous."

# We can also overwrite individual variables with the `load_data` function
iv_data.load_data(X_c = data[["new", "continuous", "cols"]])
```

## QPS Estimation 
The main QPS estimation functions are `estimate_qps`, `estimate_qps_with_decision_function`, and `estimate_qps_user_defined`, each serving different algorithmic use-cases. `estimate_qps` serves the case when the immediate output of an ONNX model serves as the treatment recommendation. `estimate_qps_with_decision_function` serves the case when an additional decision function is passed to process the ML outputs. `estimate_qps_user_defined` serves the case when the user has a custom function that outputs treatment recommendations. In general, all the functions require as input `X` an IVEstimatorDataset, `S` the number of draws per estimate, and `delta` the radius of the ball. Please refer to the documentation for the full list of keyword arguments.

```python
import pandas as pd
import numpy as np
from mlisne.qps import estimate_qps

S = 100
delta = 0.8
seed = 1 
ml_path = "path_to_your_onnx_model.onnx"

# `seed` sets np.random.seed 
qps = estimate_qps(iv_data, ml_path, S, delta, seed)
qps2 = estimate_qps(iv_data, ml_path, S, delta, seed)
assert qps == qps2

# We can specify np types for coercion if the ONNX model expects different types 
qps = estimate_qps(iv_data, ml_path, S, delta, types=(np.float64,))

# If the ONNX model takes separate continuous and discrete inputs, then we need to specify the input type and input names
qps = estimate_qps(iv_data, ml_path, S, delta, input_type=2, input_names=("c_inputs", "d_inputs"))

### QPS estimation with passing ML outputs into a decision function
from mlisne.qps import estimate_qps_with_decision_function

# We can pass the base function `round` directly into the qps estimation, which will vectorize the function for us and round the ML outputs
qps = estimate_qps_with_decision_function(iv_data, ml_path, S, delta, fcn = round)

# Additional keyword argument will be passed directly into the decision function
qps = estimate_qps_with_decision_function(iv_data, ml_path, S, delta, fcn = round, digits=5)

# We can also pass a vectorized function with the flag `vectorized` 
qps = estimate_qps_with_decision_function(iv_data, ml_path, S, delta, fcn = np.round, vectorized=True)

### QPS estimation with a user-defined function
model = pickle.load(open("path_to_your_model.pickle", 'rb'))

# Basic decision function: assign treatment if prediction > c
def assign_cutoff(X, c):
    return (X > c).astype("int")

# User-defined function to assign treatment recommendation
def ml_round(X, **kwargs):
    preds = model.predict_proba(X)
    treat = assign_cutoff(preds, **kwargs)
    return treat

qps = estimate_qps_user_defined(iris_dataset_discrete, ml_round, c = 0.5)

```

## IV Estimation
Once the QPS is estimated for each observation, the IV approach allows us to estimate the historical LATE. The TreatmentIVEstimator applies the 2SLS method to fit the model. Post-estimation diagnostics and statistics are accessible directly from the estimator. Please see the documentation for the full list of available statistics.
```python
import pandas as pd
import numpy as np
from mlisne.estimator import TreatmentIVEstimator

est = TreatmentIVEstimator()
est.fit(iv_data, qps)
print(est) 

# If we know that ML takes only one nondegenerate value (strictly between 0 and 1) in the sample, then the constant term will need to be removed
est.fit(iv_data, qps, single_nondegen=True)

# Standard statistics 
est.coef
est.std_err
est.fitted

# Post-estimation
postest = est.postest
postest['rss']
postest['r2']

# First stage statistics
fs = est.firststage
fs['coef']
fs['r2']
fs['std_error']
```
## Model Conversion
The mlisne API offers an ONNX conversion function `convert_to_onnx` that generalizes the conversion process. The function requires a dummy input to infer the input dtype, allows for renaming of input nodes, and passes downstream any framework specific keyword arguments.
```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LogisticRegression()
clr.fit(X_train, y_train)

from mlisne.helpers import convert_to_onnx

X_dummy = X[0,:]
filename = "save_path_to_onnx.onnx"

convert_to_onnx(model = model, dummy_input = X_dummy, path = filename, framework = "sklearn")

# Set custom input node name and pass additional keyword arguments
convert_to_onnx(model=model, dummy_input=X_dummy, path=filename, framework="sklearn", input_names=("input",),
                target_opset=12, doc_string="Sklearn LogisticRegression model trained on iris dataset")
```

## Further examples
- [Sklearn: model training, conversion, data generation, and estimation](https://github.com/factoryofthesun/mlisne/blob/master/examples/Sklearn_Iris_Conversion_Simulation_and_Estimation.ipynb)
- [Pytorch: neural network with categorical embeddings](https://github.com/factoryofthesun/mlisne/blob/master/examples/Pytorch_Churn_Categorical_Embeddings.ipynb)

# Versioning 

# Contributing

# Citation

# License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

# Authors

# Acknowledgements

## References

## Related Projects

