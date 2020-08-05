# Machine Learning is Natural Experiment
**[Overview](#overview)** | **[Installation](#installation)** | **[Usage](#usage)** | **[References](#references)**
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

insert images here...

## MLisNE Package

The mlisne package is an implementation of the treatment effect estimation method described above. This package provides a simple-to-use pipeline for data preprocessing, QPS estimation, and treatment effect estimation that is ML framework-agnostic. 

### Supported ML Frameworks

The QPS estimation function `estimate_qps` only accepts models in the ONNX framework in order to maintain the framework-agnostic implementation. The module provides a `convert_to_onnx` function that currently supports conversion from the following frameworks

- [Sklearn](https://github.com/onnx/sklearn-onnx/)
- [Pytorch](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

For conversion functions for other frameworks, please refer to the [onnxmltools repository](https://github.com/onnx/onnxmltools).

# Installation 
You can install OBP using Python's package manager `pip`.

```
pip install obp
```

You can install OBP from source.
```bash
git clone https://github.com/factoryofthesun/mlisne
cd mlisne
python setup.py install
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
# Versioning 

# Contributing

# Citation

# License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

# Authors

# Acknowledgements

## References

## Related Projects

