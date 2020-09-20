.. mlisne documentation master file, created by
   sphinx-quickstart on Wed Aug 26 15:16:03 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLISNE: Machine Learning is Natural Experiment Documentation
============================================================

Overview
~~~~~~~~

The mlisne package is a generic implementation of the treatment effect estimation method proposed in :cite:`Narita2020`. The package provides functions for QPS estimation and treatment effect estimation, and is designed to be flexible to the researcher's specific treatment interface.

Supported ML Frameworks
~~~~~~~~~~~~~~~~~~~~~~~

The ML-agnostic QPS estimation function ``estimate_qps_onnx`` only accepts models in the ONNX framework in order to maintain the generalized implementation. The module provides an ONNX conversion function ``convert_to_onnx`` that currently supports conversion from the following frameworks:

- `Sklearn <https://github.com/onnx/sklearn-onnx/>`_
- `Pytorch <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>`_
- `LightGBM <https://github.com/microsoft/LightGBM>`_
- `XGBoost (experimental) <https://github.com/dmlc/xgboost>`_
- `CatBoost (experimental) <https://github.com/catboost/catboost>`_
- `CoreML (experimental) <https://github.com/apple/coremltools>`_
- `LibSVM (experimental) <https://github.com/cjlin1/libsvm>`_
- `SparkML (experimental) <https://spark.apache.org/mllib/>`_
- `Keras (experimental) <https://keras.io/>`_

For conversion functions for other frameworks, please refer to the `onnxmltools repository <https://github.com/onnx/onnxmltools>`_.
Please note that use of these functions requires that the relevant framework packages are installed.

License
~~~~~~~

This project is licensed under the Apache 2.0 License - see the `LICENSE <https://github.com/factoryofthesun/mlisne/blob/master/LICENSE>`_ file for details.

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   about
   method

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   mlisne

.. toctree::
   :maxdepth: 2
   :caption: Others:

   references
   Github <https://github.com/factoryofthesun/mlisne>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
