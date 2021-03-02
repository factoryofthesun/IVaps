Quickstart
==========

Please see below for a minimal example of the IVaps procedure to estimate LATE, given a converted ONNX model and historical treatment data.

.. code-block:: python

  import pandas as pd
  import numpy as np
  import onnxruntime as rt
  from IVaps import estimate_aps_onnx
  from IVaps import estimate_treatment_effect

  # Read in data
  data = pd.read_csv("path_to_your_historical_data.csv")

  # Estimate APS with 100 draws and ball radius 0.8
  aps = estimate_aps_onnx(X_c = data[["names", "of", "continuous", "variables"]], X_d = data[["names", "of", "discrete", "variables"]], S=100, delta=0.8, ML_onnx="path_to_onnx_model.onnx")

  # Fit 2SLS Estimation model
  model = estimate_treatment_effect(Y = data["outcome"], Z = data["treatment_recommendation"], D = data["treatment_assignment"], aps = aps)

  # Prints estimation results
  print(model) # Print summary of second stage results
  print(model.first_stage) # Print summary of first stage results
  print(model.first_stage.individual['D']) # Another view of first stage results

  model.params # Array of second-stage estimated coefficients
  model.cov # Variance covariance matrix

The following is a breakdown of each of the two main estimation steps along with how the package handles different user cases.

APS Estimation
~~~~~~~~~~~~~~

The main APS estimation functions are ``estimate_aps_onnx``, and ``estimate_aps_user_defined``, each serving different algorithmic use-cases. ``estimate_aps_onnx`` serves the case when an ONNX model with an optional post-prediction decision function produces the treatment recommendation. ``estimate_aps_user_defined`` serves the case when the user has a custom function that takes an array of ML inputs and outputs treatment recommendations. APS estimation requires at minimum that we have ``X_c`` a collection of continuous variable data, ``S`` the number of draws per estimate, and ``delta`` the radius of the ball. As will be demonstrated below, however, there are a number of different ways that users can pass in data for estimation. Please refer to the documentation for the full list of keyword parameters and their default values.

**Note:** ``data``, ``X_c``, and ``X_d`` should never have any overlapping columns. This is impossible to check in the code, so best practice is to use either ``data`` and index inputs ``C`` and ``D`` or ``X_c`` and ``X_d`` separately.

.. code-block:: python

  import pandas as pd
  import numpy as np
  from IVaps import estimate_aps_onnx

  S = 100
  delta = 0.8
  seed = 1 # `seed` sets np.random.seed
  ml_path = "path_to_your_onnx_model.onnx"
  data = pd.read_csv("path_to_your_historical_data.csv")

  ### The below function calls will all output the same results
  aps0 = estimate_aps_onnx(ml_path, X_c = data[['continuous', 'variables']], X_d = data[['discrete', 'variables']], S = 100, delta = 0.8, seed = seed)
  aps1 = estimate_aps_onnx(ml_path, data = data, C = indices_of_cts_vars, D = indices_of_discrete_vars, S = 100, delta = 0.8, seed = seed)

  # The function infers data greedily, so that whatever variables are not explicitly passed will be inferred from the remaining data
  aps2 = estimate_aps_onnx(ml_path, data = data[['continuous', 'vars']], X_d = data[['discrete', 'vars']], S = 100, delta = 0.8, seed = seed) # Assumes all of `data` is continuous
  aps3 = estimate_aps_onnx(ml_path, data = data[['discrete', 'vars']], X_c = data[['continuous', 'vars']], S = 100, delta = 0.8, seed = seed) # Assumes all of `data` is discrete
  aps4 = estimate_aps_onnx(ml_path, data = data[['cts', 'and', 'discrete', 'vars']], C = indices_of_cts_vars, S = 100, delta = 0.8, seed = seed) # Assumes remaining columns of `data` are discrete
  aps5 = estimate_aps_onnx(ml_path, data = data[['cts', 'and', 'discrete', 'vars']], D = indices_of_discrete_vars, S = 100, delta = 0.8, seed = seed) # Assumes remaining columns of `data` are continuous

  assert aps0 == aps1 == aps2 == aps3 == aps4 == aps5

  # If only the data object is passed, then all variables will be assumed to be continuous
  aps = estimate_aps_onnx(ml_path, data = data[['all', 'continuous', 'variables']], S = 100, delta = 0.8,)

  # We can specify np types for coercion if the ONNX model expects different types
  aps = estimate_aps_onnx(ml_path, data = data, S = 100, delta = 0.8, types=(np.float64,))

  # If the ONNX model takes separate continuous and discrete inputs, then we need to specify the input type and input names
  aps = estimate_aps_onnx(ml_path, data = data, S = 100, delta = 0.8, input_type=2, input_names=("c_inputs", "d_inputs"))

  ### APS estimation with passing ML outputs into a decision function

  # We can pass the base function `round` directly into the aps estimation, which will vectorize the function for us and round the ML outputs
  aps = estimate_aps_onnx(ml_path, data = data, S = 100, delta = 0.8, fcn = round)

  # Additional keyword argument will be passed directly into the decision function
  aps = estimate_aps_onnx(ml_path, data = data, S = 100, delta = 0.8, fcn = round, digits=5)

  # We can also pass a vectorized function with the flag `vectorized`
  aps = estimate_aps_onnx(ml_path, data = data, S = 100, delta = 0.8, fcn = np.round, vectorized=True)

  ### APS estimation with a user-defined function
  from IVaps import estimate_aps_user_defined

  model = pickle.load(open("path_to_your_model.pickle", 'rb'))

  # Basic decision function: assign treatment if prediction > c
  def assign_cutoff(X, c):
      return (X > c).astype("int")

  # User-defined function to assign treatment recommendation
  def ml_round(X, **kwargs):
      preds = model.predict_proba(X)
      treat = assign_cutoff(preds, **kwargs)
      return treat

  aps = estimate_aps_user_defined(data = data, ml = ml_round, c = 0.5)

  # We can parallelize APS computation with a user defined function -- see documentation for more details
  aps = estimate_aps_user_defined(data = data, ml = ml_round, c = 0.5, parallel = True)

Multiple Delta Estimation
-------------------------

In the case that the researcher wishes to estimate APS using different sized deltas without having to call APS estimation multiple times, both APS estimation functions can take a list of delta values. In this case the same draws from a standard uniform ball will be used, scaled up by the different delta radii. The output APS array will have shape (# observations, # deltas), with each column corresponding to the respective delta in the order of the input list.

.. code-block:: python

  import pandas as pd
  import numpy as np
  from IVaps import estimate_aps_onnx

  delta_list = [0.1, 0.5, 0.8]
  ml_path = "path_to_your_onnx_model.onnx"
  data = pd.read_csv("path_to_your_historical_data.csv")

  ### The below function calls will all output the same results
  aps = estimate_aps_onnx(ml_path, X_c = data[['continuous', 'variables']], X_d = data[['discrete', 'variables']], S = 100, delta = delta_list)

  assert aps.shape[1] == len(delta_list)

Mixed Variables and Missing Values Treatment
--------------------------------------------

APS estimation is also equipped to handle mixed variables (variables that have both a discrete and continuous part), and will treat mixed variables as a subset of the continuous variables. The user will need to pass a dictionary ``L``, where the keys are the indices of ``X_c`` that are mixed, and the values are sets of the discrete values each variable takes on. During estimation, if an observation of a continuous variable equals any of its discrete parts, then it will be treated as a discrete variable for that observation. Similarly, if the function encounters an observation of a missing value, then the variable will be assumed to be discrete for that sample observation.

.. code-block:: python

  import pandas as pd
  import numpy as np
  from IVaps import estimate_aps_onnx

  data_with_missing = pd.read_csv("path_to_your_historical_data_with_missing.csv")
  ml_path = "path_to_your_onnx_model.onnx"

  # Create mixed variables dictionary
  L = {0: {0}, 3: {5, 10}} # This indicates that the 0th and 3rd index continuous variables are mixed variables with the passed discrete parts

  # APS estimation
  aps = estimate_aps_onnx(ml_path, data = data_with_missing, L = L)

Pandas Compatibility
--------------------

Sometimes the custom user function may require a pandas dataframe as an input and be column-order or column-name sensitive. Below are examples of how to pass these options into APS estimation.

.. code-block:: python

  import pandas as import pd
  from IVaps import estimate_aps_user_defined

  data = pd.read_csv("path_to_your_historical_data.csv")

  # If the custom function expects pandas data, we need to set the `pandas` flag and optionally assign column names
  aps = estimate_aps_user_defined(data = data, ml = pandas_dependent_function, pandas = True, pandas_cols = ['list', 'of', 'column', 'names'])

  # We can also have the inputs maintain the original order that we passed them in
  aps = estimate_aps_user_defined(data = data, ml = function_that_needs_original_column_ordering, pandas = True, pandas_cols = ['list', 'of', 'column', 'names'], keep_order = True)

  # We can also do a custom reordering of the columns -- the arguments `keep_order`, `reorder`, and `pandas_cols` are applied sequentially in that order
  # The below example will apply the ordering using the indices passed into `reorder` onto the original column order
  aps = estimate_aps_user_defined(data = data, ml = function_that_needs_new_column_ordering, pandas = True, pandas_cols = ['list', 'of', 'column', 'names'], keep_order = True, reorder = new_ordering)

  # The below example will apply the reordering on the default input order, which is [continuous_variables, discrete_variables]
  aps = estimate_aps_user_defined(data = data, ml = function_that_needs_new_column_ordering, pandas = True, pandas_cols = ['list', 'of', 'column', 'names'], reorder = new_ordering)

IV Estimation
~~~~~~~~~~~~~

Once the APS is estimated for each observation, the IV approach allows us to estimate the historical LATE. ``estimate_treatment_effect`` is our primary IV estimation function, and makes use of the IV2SLS class from the `linearmodels package <https://bashtage.github.io/linearmodels/>`_. As per the package, the function will return an IVResults object. Post-estimation diagnostics and statistics are accessible directly from this object. Please refer to the `object documentation <https://bashtage.github.io/linearmodels/doc/iv/results.html#linearmodels.iv.results.IVResults>`_ for a full list of accessible attributes.

**Note:** Similar to APS estimation, the inputs can be passed in through a combination of a ``data`` object or ``aps``, ``Y``, ``Z``, and ``D`` separately, but they must not have any overlapping columns. Recommended best practice is to use either ``data`` and index inputs ``aps_ind``, ``Y_ind``, ``Z_ind`` and ``D_ind`` or ``aps``, ``Y``, ``Z``, and ``D`` separately.

.. code-block:: python

  import pandas as pd
  import numpy as np
  from IVaps import estimate_treatment_effect

  model = estimate_treatment_effect(Y = outcome_variable, Z = treatment_recommendation, D = treatment_assignment, aps = estimated_aps, verbose = False)
  print(model)

  # If we know that ML takes only one nondegenerate value (strictly between 0 and 1) in the sample, then the constant term will need to be removed by setting single_nondegen
  model = estimate_treatment_effect(Y = outcome_variable, Z = treatment_recommendation, D = treatment_assignment, aps = estimated_aps, single_nondegen = True)

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

Counterfactual Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~

Counterfactual ML value estimation is provided through the ``estimate_counterfactual_ml`` function. The function fits an OLS regression of outcomes on treatment recommendation controlling for APS, then uses the estimated effect of recommendation to estimate counterfactual outcomes of a different recommendation system.

.. code-block:: python

  import pandas as pd
  from IVaps import estimate_counterfactual_ml, estimate_aps_onnx

  data = pd.read_csv("path_to_your_historical_data.csv")
  ml_path = "path_to_onnx_model.onnx"
  aps = estimate_aps_onnx(ml_path, data[['cts', 'vars']], data[['discrete', 'vars']])

  original_ml_recs = pd.read_csv("original_ml_recs.csv")
  counterfactual_ml_recs = pd.read_csv("counterfactual_ml_recs.csv")

  cf_values, ols_model = estimate_counterfactual_ml(Y = data['Y'], Z = data['Z'], aps = aps, recs = original_ml_recs, cf_recs = counterfactual_ml_recs, verbose = True)

  mean_counterfactual_value = cf_values.mean()

Model Conversion
~~~~~~~~~~~~~~~~

The IVaps API offers an ONNX conversion function ``convert_to_onnx`` that generalizes the conversion process by wrapping the conversion functions offered by `onnxmltools <https://github.com/onnx/onnxmltools>`_. The function requires a dummy input to infer the input dtype, allows for renaming of input nodes, and passes downstream any framework specific keyword arguments.

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

  from IVaps.helpers import convert_to_onnx

  X_dummy = X[0,:]
  filename = "save_path_to_onnx.onnx"

  convert_to_onnx(model = model, dummy_input = X_dummy, path = filename, framework = "sklearn")

  # Set custom input node name and pass additional keyword arguments
  convert_to_onnx(model=model, dummy_input=X_dummy, path=filename, framework="sklearn", input_names=("input",),
                  target_opset=12, doc_string="Sklearn LogisticRegression model trained on iris dataset")
