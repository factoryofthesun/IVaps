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

Today’s society increasingly resorts to machine learning (“AI”) and other algorithms for decision-making and resource allocation. For example, judges make legal judgements using predictions from supervised machine learning (descriptive regression). Supervised learning is also used by governments to detect potential criminals and terrorists, and financial companies (such as banks and insurance companies) to screen potential customers. Tech companies like Facebook, Microsoft, and Netflix allocate digital content by reinforcement learning and bandit algorithms. Uber and other ride sharing services adjust prices using their surge pricing algorithms to take into account local demand and supply information. Retailers and e-commerce platforms like Amazon engage in algorithmic pricing. Similar algorithms are invading into more and more high-stakes treatment assignment, such as education, health, and military.

All of the above, seemingly diverse examples share a common trait: An algorithm makes decisions based only on observable input variables the data-generating algorithm uses. Conditional on the observable variables, therefore, algorithmic treatment decisions are (quasi-)randomly assigned. This property makes algorithm-based treatment decisions **an instrumental variable we can use for measuring the causal effect of the final treatment assignment**. The algorithm-based instrument may produce regression-discontinuity-style local variation (e.g. machine judges), stratified randomization (e.g. several bandit and reinforcement leaning algorithms), or mixes of the two. Narita 2020 introduces the formal framework and characterizes the sources of causal effect identification. [[1]](#1)

## Framework
<img src="/images/ml_natural_experiment_diagram.PNG" width="550" height="250"/>
<img src="/images/framework_1.PNG" width="500" height="300"/>
<img src="/images/framework_2.PNG" width="500" height="300"/>
<img src="/images/framework_3.PNG" width="500" height="300"/>
<img src="/images/qps_chart.PNG" width="350" height="250"/>
<img src="/images/qps_estimation.PNG" width="500" height="300"/>

## MLisNE Package

The mlisne package is an implementation of the treatment effect estimation method and paper described above. This package provides functions for the two primary estimation steps -- QPS estimation and treatment effect estimation -- and is ML framework-agnostic.

### Supported ML Frameworks

The QPS estimation function for trained models `estimate_qps_onnx` only accepts models in the ONNX framework in order to maintain the framework-agnostic implementation. The module provides a `convert_to_onnx` function that currently supports conversion from the following frameworks:

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
The installation will automatically detect whether there is a compatible GPU device on the system and install either onnxruntime or onnxruntime-gpu. Please note that the default onnxruntime GPU build requires CUDA runtime libraries being installed on the system. Please see the [onnxruntime repository](https://github.com/microsoft/onnxruntime) for more details regarding the GPU build. 

# Requirements

# Usage
Below is a general example of how to use the modules in this package to estimate LATE, given a converted ONNX model and historical treatment data.
```python
import pandas as pd
import numpy as np
import onnxruntime as rt
from mlisne import estimate_qps_onnx
from mlisne import estimate_treatment_effect

# Read in data
data = pd.read_csv("path_to_your_historical_data.csv")

# Estimate QPS with 100 draws and ball radius 0.8
qps = estimate_qps_onnx(X_c = data[["names", "of", "continuous", "variables"]], X_d = data[["names", "of", "discrete", "variables"]], S=100, delta=0.8, ML_onnx="path_to_onnx_model.onnx")

# Fit 2SLS Estimation model
model = estimate_treatment_effect(Y = data["outcome"], Z = data["treatment_recommendation"], D = data["treatment_assignment"], qps = qps)

# Prints estimation results
print(model) # Print summary of second stage results
print(model.first_stage) # Print summary of first stage results
print(model.first_stage.individual['D']) # Another view of first stage results

model.params # Array of second-stage estimated coefficients
model.cov # Variance covariance matrix
```
In general the MLisNE method has two main estimation steps: QPS estimation and IV estimation.

## QPS Estimation
The main QPS estimation functions are `estimate_qps_onnx`, and `estimate_qps_user_defined`, each serving different algorithmic use-cases. `estimate_qps_onnx` serves the case when an ONNX model with an optional post-prediction decision function produces the treatment recommendation. `estimate_qps_user_defined` serves the case when the user has a custom function that takes an array of ML inputs and outputs treatment recommendations. QPS estimation requires at minimum that we have `X_c` a collection of continuous variable data, `S` the number of draws per estimate, and `delta` the radius of the ball. As will be demonstrated below, however, there are a number of different ways that users can pass in data for estimation. Please refer to the documentation for the full list of keyword parameters and their default values.

```python
import pandas as pd
import numpy as np
from mlisne import estimate_qps_onnx

S = 100
delta = 0.8
seed = 1 # `seed` sets np.random.seed
ml_path = "path_to_your_onnx_model.onnx"
data = pd.read_csv("path_to_your_historical_data.csv")

### The below function calls will all output the same results
qps0 = estimate_qps_onnx(ml_path, X_c = data[['continuous', 'variables']], X_d = data[['discrete', 'variables']], S = 100, delta = 0.8, seed = seed)
qps1 = estimate_qps_onnx(ml_path, data = data, C = indices_of_cts_vars, D = indices_of_discrete_vars, S = 100, delta = 0.8, seed = seed)

# The function infers data greedily, so that whatever variables are not explicitly passed will be inferred from the remaining data
qps2 = estimate_qps_onnx(ml_path, data = data[['continuous', 'vars']], X_d = data[['discrete', 'vars']], S = 100, delta = 0.8, seed = seed) # Assumes all of `data` is continuous
qps3 = estimate_qps_onnx(ml_path, data = data[['discrete', 'vars']], X_c = data[['continuous', 'vars']], S = 100, delta = 0.8, seed = seed) # Assumes all of `data` is discrete
qps4 = estimate_qps_onnx(ml_path, data = data[['cts', 'and', 'discrete', 'vars']], C = indices_of_cts_vars, S = 100, delta = 0.8, seed = seed) # Assumes remaining columns of `data` are discrete
qps5 = estimate_qps_onnx(ml_path, data = data[['cts', 'and', 'discrete', 'vars']], D = indices_of_discrete_vars, S = 100, delta = 0.8, seed = seed) # Assumes remaining columns of `data` are continuous

assert qps0 == qps1 == qps2 == qps3 == qps4 == qps5

# If only the data object is passed, then all variables will be assumed to be continuous
qps = estimate_qps_onnx(ml_path, data = data[['all', 'continuous', 'variables']], S = 100, delta = 0.8,)

# We can specify np types for coercion if the ONNX model expects different types
qps = estimate_qps_onnx(ml_path, data = data, S = 100, delta = 0.8, types=(np.float64,))

# If the ONNX model takes separate continuous and discrete inputs, then we need to specify the input type and input names
qps = estimate_qps_onnx(ml_path, data = data, S = 100, delta = 0.8, input_type=2, input_names=("c_inputs", "d_inputs"))

### QPS estimation with passing ML outputs into a decision function

# We can pass the base function `round` directly into the qps estimation, which will vectorize the function for us and round the ML outputs
qps = estimate_qps_onnx(ml_path, data = data, S = 100, delta = 0.8, fcn = round)

# Additional keyword argument will be passed directly into the decision function
qps = estimate_qps_onnx(ml_path, data = data, S = 100, delta = 0.8, fcn = round, digits=5)

# We can also pass a vectorized function with the flag `vectorized`
qps = estimate_qps_onnx(ml_path, data = data, S = 100, delta = 0.8, fcn = np.round, vectorized=True)

### QPS estimation with a user-defined function
from mlisne import estimate_qps_user_defined

model = pickle.load(open("path_to_your_model.pickle", 'rb'))

# Basic decision function: assign treatment if prediction > c
def assign_cutoff(X, c):
    return (X > c).astype("int")

# User-defined function to assign treatment recommendation
def ml_round(X, **kwargs):
    preds = model.predict_proba(X)
    treat = assign_cutoff(preds, **kwargs)
    return treat

qps = estimate_qps_user_defined(data = data, ml = ml_round, c = 0.5)
```

### Pandas Compatibility
Sometimes the custom user function may require a pandas dataframe as an input and be column-order or column-name sensitive. Below are examples of how to pass these options into QPS estimation.
```python
import pandas as import pd
from mlisne import estimate_qps_user_defined

data = pd.read_csv("path_to_your_historical_data.csv")

# If the custom function expects pandas data, we need to set the `pandas` flag and optionally assign column names
qps = estimate_qps_user_defined(data = data, ml = pandas_dependent_function, pandas = True, pandas_cols = ['list', 'of', 'column', 'names'])

# We can also have the inputs maintain the original order that we passed them in
qps = estimate_qps_user_defined(data = data, ml = function_that_needs_original_column_ordering, pandas = True, pandas_cols = ['list', 'of', 'column', 'names'], keep_order = True)

# We can also do a custom reordering of the columns -- the arguments `keep_order`, `reorder`, and `pandas_cols` are applied sequentially in that order
# The below example will apply the ordering using the indices passed into `reorder` onto the original column order
qps = estimate_qps_user_defined(data = data, ml = function_that_needs_new_column_ordering, pandas = True, pandas_cols = ['list', 'of', 'column', 'names'], keep_order = True, reorder = new_ordering)

# The below example will apply the reordering on the default input order, which is [continuous_variables, discrete_variables]
qps = estimate_qps_user_defined(data = data, ml = function_that_needs_new_column_ordering, pandas = True, pandas_cols = ['list', 'of', 'column', 'names'], reorder = new_ordering)
```

### Mixed Variables and Missing Values Treatment
QPS estimation is also equipped to handle mixed variables (variables that have both a discrete and continuous part), and will treat mixed variables as a subset of the continuous variables. The user will need to pass a dictionary ``L``, where the keys are the indices of ``X_c`` that are mixed, and the values are sets of the discrete values each variable takes on. During estimation, if an observation of a continuous variable equals any of its discrete parts, then it will be treated as a discrete variable for that observation. Similarly, if the function encounters an observation of a missing value, then the variable will be assumed to be discrete for that sample observation.

```python
import pandas as pd
import numpy as np
from mlisne import estimate_qps_onnx

data_with_missing = pd.read_csv("path_to_your_historical_data_with_missing.csv")
ml_path = "path_to_your_onnx_model.onnx"

# Create mixed variables dictionary
L = {0: {0}, 3: {5, 10}} # This indicates that the 0th and 3rd index continuous variables are mixed variables with the passed discrete parts

# QPS estimation
qps = estimate_qps_onnx(ml_path, data = data_with_missing, L = L)
```

## IV Estimation
Once the QPS is estimated for each observation, the IV approach allows us to estimate the historical LATE. `estimate_treatment_effect` is our primary IV estimation function, and makes use of the IV2SLS class from the [linearmodels package](https://bashtage.github.io/linearmodels/). As per the package, the function will return an IVResults object. Post-estimation diagnostics and statistics are accessible directly from this object. Please refer to the [object documentation](https://bashtage.github.io/linearmodels/doc/iv/results.html#linearmodels.iv.results.IVResults) for a full list of accessible attributes.

```python
import pandas as pd
import numpy as np
from mlisne import estimate_treatment_effect

model = estimate_treatment_effect(Y = outcome_variable, Z = treatment_recommendation, D = treatment_assignment, qps = estimated_qps, verbose = False)
print(model)

# If we know that ML takes only one nondegenerate value (strictly between 0 and 1) in the sample, then the constant term will need to be removed by setting single_nondegen
model = estimate_treatment_effect(Y = outcome_variable, Z = treatment_recommendation, D = treatment_assignment, qps = estimated_qps, single_nondegen = True)

# Standard statistics
model.params
model.cov
model.std_errors
model.fitted_values
model.rsquared
model.model_ss # residual sum of squares

# First stage statistics
print(model.first_stage)
fs = model.first_stage.individual['D']
fs.params
fs.rsquared
fs.std_errors
```

## Counterfactual Estimation
Counterfactual ML value estimation is provided through the `estimate_counterfactual_ml` function. The function fits an OLS regression of outcomes on treatment recommendation controlling for QPS, then uses the estimated effect of recommendation to estimate counterfactual outcomes of a different recommendation system.

```python
import pandas as pd
from mlisne import estimate_counterfactual_ml, estimate_qps_onnx

data = pd.read_csv("path_to_your_historical_data.csv")
ml_path = "path_to_onnx_model.onnx"
qps = estimate_qps_onnx(ml_path, data[['cts', 'vars']], data[['discrete', 'vars']])

original_ml_recs = pd.read_csv("original_ml_recs.csv")
counterfactual_ml_recs = pd.read_csv("counterfactual_ml_recs.csv")

cf_values, ols_model = estimate_counterfactual_ml(Y = data['Y'], Z = data['Z'], qps = qps, recs = original_ml_recs, cf_recs = counterfactual_ml_recs, verbose = True)

mean_counterfactual_value = cf_values.mean()
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
<a id="1">[1]</a>
1. Narita, Yusuke and Yata, Kohei. Machine Learning is Natural Experiment (forthcoming). 2020.

## Related Projects
