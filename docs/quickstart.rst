Quickstart
==========

Please see below for a minimal example of the mlisne procedure to estimate LATE, given a converted ONNX model and historical treatment data.

.. code-block:: python

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
  print(estimator) # Print summary of second stage results
  estimator.firststage_summary() # Print summary of first stage results

  estimator.coef # Array of second-stage estimated coefficients
  estimator.varcov # Variance covariance matrix

The following is a breakdown of the different pipeline steps along with how the package handles different user cases.

Data Loading
~~~~~~~~~~~~

The IVEstimatorDataset class is the main data loader for the rest of the pipeline. It splits the data into individual arrays of the outcome `Y``, treatment assignment ``D``, algorithmic recommendation ``Z``, continuous inputs ``X_c``, and discrete inputs ``X_d``. The module can be initialized by passing in a pandas dataframe or numpy array with associated indices, or the variables can be individually assigned.

.. code-block:: python

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

Mixed Variables Treatment
-------------------------

The data loader is also equipped to handle mixed variables (variables that have both a discrete and continuous part), and will treat mixed variables as a subset of the continuous variables. The dataset stores an object ``L``, which is a dictionary where the keys are the indices of ``X_c`` that are mixed, and the values are sets of the discrete values each variable takes on. The rest of the pipeline operates as described below irrespective of mixed variables.

.. code-block:: python

  import pandas as pd
  import numpy as np
  from mlisne.dataset import IVEstimatorDataset

  data = pd.read_csv("path_to_your_historical_data.csv")

  # Create mixed variables dictionary
  L = {0: {0}, 3: {5, 10}} # This indicates that the 0th and 3rd index continuous variables are mixed variables with the passed discrete parts

  # Initialization
  iv_data = IVEstimatorDataset(data, L = L)

  # L can also be assigned directly
  iv_data.L = L


QPS Estimation
~~~~~~~~~~~~~~

The main QPS estimation functions are ``estimate_qps``, ``estimate_qps_with_decision_function``, and ``estimate_qps_user_defined``, each serving different algorithmic use-cases. ``estimate_qps`` serves the case when the immediate output of an ONNX model serves as the treatment recommendation. ``estimate_qps_with_decision_function`` serves the case when an additional decision function is passed to process the ML outputs. ``estimate_qps_user_defined`` serves the case when the user has a custom function that outputs treatment recommendations. In general, all the functions require as input ``X`` an IVEstimatorDataset, ``S`` the number of draws per estimate, and ``delta`` the radius of the ball. Please refer to the documentation for the full list of keyword arguments.

.. code-block:: python

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

IV Estimation
~~~~~~~~~~~~~

Once the QPS is estimated for each observation, the IV approach allows us to estimate the historical LATE. The TreatmentIVEstimator applies the 2SLS method to fit the model. Post-estimation diagnostics and statistics are accessible directly from the estimator. Please see the documentation for the full list of available statistics.

.. code-block:: python

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

Model Conversion
~~~~~~~~~~~~~~~~

The mlisne API offers an ONNX conversion function ``convert_to_onnx`` that generalizes the conversion process. The function requires a dummy input to infer the input dtype, allows for renaming of input nodes, and passes downstream any framework specific keyword arguments.

.. code-block:: python

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
