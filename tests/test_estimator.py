# Test IV Estimator
import sys
import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import onnxruntime as rt
from pathlib import Path
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

from mlisne.dataset import EstimatorDataset
from mlisne.qps import estimate_qps
from mlisne.estimator import TreatmentEffectsEstimator, CounterfactualMLEstimator

sklearn_logreg = str(Path(__file__).resolve().parents[0] / "test_models" / "logreg_iris.onnx")
sklearn_logreg_double = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_double.onnx")
sklearn_logreg_infer = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_infertype.onnx")
model_path = str(Path(__file__).resolve().parents[1] / "examples" / "models")
data_path = str(Path(__file__).resolve().parents[1] / "examples" / "data")

@pytest.fixture
def empty_estimator():
    est = TreatmentEffectsEstimator()
    return est

@pytest.fixture
def empty_cf_estimator():
    est = CounterfactualMLEstimator()
    return est

@pytest.fixture
def iris_data():
    iris = pd.read_csv(f"{data_path}/iris_data.csv")
    return iris

def test_all_empty(empty_estimator):
    assert empty_estimator.coef is None
    assert empty_estimator.varcov is None
    assert empty_estimator.fitted is None
    assert empty_estimator.resid is None
    assert empty_estimator.tstat is None
    assert empty_estimator.std_error is None
    assert empty_estimator.p is None
    assert empty_estimator.n_fit is None
    assert empty_estimator.ci is None
    assert empty_estimator.coef is None
    assert empty_estimator.firststage is None

def test_iris_estimation(empty_estimator, iris_data):
    data = np.array(iris_data.drop(['Y0', 'Y1'], axis=1))
    dataset = EstimatorDataset(data)
    qps = estimate_qps(dataset, f"{model_path}/iris_logreg.onnx")
    empty_estimator.fit(dataset, qps)

    print(empty_estimator) # Should print summary table

    # Test second stage values
    coef = empty_estimator.coef
    N = empty_estimator.n_fit
    resid = empty_estimator.resid
    fitted = empty_estimator.fitted
    std_error = empty_estimator.std_error
    p_ss = empty_estimator.p
    t_ss = empty_estimator.tstat
    ci_ss = empty_estimator.ci
    adj_inps = empty_estimator.inputs

    postest = empty_estimator.postest
    r2_ss = postest['r2']

    exog = sm.add_constant(adj_inps['QPS'])
    iv_check = IV2SLS(adj_inps['Y'], exog, adj_inps['D'], adj_inps['Z']).fit(cov_type='robust')
    validation_ci = iv_check.conf_int()
    validation_stderr = iv_check.std_errors
    validation_coef = iv_check.params
    validation_fitted = iv_check.fitted_values
    validation_p = iv_check.pvalues
    validation_t = iv_check.tstats
    validation_r2 = iv_check.rsquared
    validation_resids = iv_check.resids
    validation_n = iv_check.nobs

    print("Debiased:", iv_check.debiased)
    assert np.array_equal(np.round(coef, 5), np.round(np.array([validation_coef['exog.0'],\
                          validation_coef['endog'], validation_coef['exog.1']]), 5))
    assert N == validation_n
    assert np.array_equal(np.round(resid, 5), np.round(validation_resids.to_numpy().flatten(), 5))
    assert np.array_equal(np.round(fitted, 5), np.round(validation_fitted.to_numpy().flatten(), 5))
    assert np.array_equal(np.round(std_error, 5), np.round(validation_stderr.to_numpy().flatten()[[0,2,1]], 5))
    assert np.array_equal(np.round(t_ss, 5), np.round(validation_t.to_numpy().flatten()[[0,2,1]], 5))
    assert np.array_equal(np.round(r2_ss, 5), np.round(validation_r2, 5))

    # Test first stage values
    firststage = empty_estimator.firststage
    coef = firststage["coef"]
    resid = firststage["resid"]
    fitted = firststage["fitted"]
    std_error = firststage["std_error"]
    p = firststage["p"]
    t = firststage["tstat"]
    ci = firststage["ci"]
    r2 = firststage["r2"]

    fs_check = iv_check.first_stage.individual['endog']
    print(fs_check)
    validation_ci = fs_check.conf_int()
    validation_stderr = fs_check.std_errors
    validation_coef = fs_check.params
    validation_fitted = fs_check.fitted_values
    validation_p = fs_check.pvalues
    validation_t = fs_check.tstats
    validation_r2 = fs_check.rsquared
    validation_resids = fs_check.resids
    validation_n = fs_check.nobs

    print("Debiased:", fs_check.debiased)
    assert np.array_equal(np.round(coef, 5), np.round(np.array([validation_coef['exog.0'],\
                          validation_coef['instruments'], validation_coef['exog.1']]), 5))
    assert N == validation_n
    assert np.array_equal(np.round(resid, 5), np.round(validation_resids.to_numpy().flatten(), 5))
    assert np.array_equal(np.round(fitted, 5), np.round(validation_fitted.to_numpy().flatten(), 5))
    assert np.array_equal(np.round(std_error, 5), np.round(validation_stderr.to_numpy().flatten()[[0,2,1]], 5))
    assert np.array_equal(np.round(t, 5), np.round(validation_t.to_numpy().flatten()[[0,2,1]], 5))
    assert np.array_equal(np.round(r2, 5), np.round(validation_r2, 5))

def test_cf_all_empty(empty_cf_estimator):
    assert empty_cf_estimator.coef is None
    assert empty_cf_estimator.varcov is None
    assert empty_cf_estimator.fitted is None
    assert empty_cf_estimator.resid is None
    assert empty_cf_estimator.tstat is None
    assert empty_cf_estimator.std_error is None
    assert empty_cf_estimator.p is None
    assert empty_cf_estimator.n_fit is None
    assert empty_cf_estimator.ci is None

def test_counterfactual_estimation(iris_data, empty_cf_estimator):
    data = np.array(iris_data.drop(['Y0', 'Y1'], axis=1))
    dataset = EstimatorDataset(data)
    qps = estimate_qps(dataset, f"{sklearn_logreg}", types=(np.float32,None))
    empty_cf_estimator.fit(dataset, qps)

    print(empty_cf_estimator)

    og_sess = rt.InferenceSession(sklearn_logreg)
    og_input = og_sess.get_inputs()[0].name
    og_preds = og_sess.run(['output_probability'], {og_input: dataset.X_c.astype(np.float32)})[0]
    og_out = np.array([d[1] for d in og_preds])

    new_sess = rt.InferenceSession(sklearn_logreg_infer)
    new_input = new_sess.get_inputs()[0].name
    new_preds = new_sess.run(['output_probability'], {new_input: dataset.X_c.astype(np.float32)})[0]
    new_out = np.array([d[1] for d in new_preds])

    pred_value = empty_cf_estimator.predict_counterfact(dataset.Y, og_out, new_out)
    validate_value = np.mean(dataset.Y + empty_cf_estimator.coef[1] * (new_out - og_out))
    assert pred_value == validate_value
