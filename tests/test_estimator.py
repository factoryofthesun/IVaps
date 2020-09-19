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

from mlisne import estimate_qps_onnx, estimate_qps_user_defined
from mlisne import estimate_treatment_effect, estimate_counterfactual_ml

sklearn_logreg = str(Path(__file__).resolve().parents[0] / "test_models" / "logreg_iris.onnx")
sklearn_logreg_double = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_double.onnx")
sklearn_logreg_infer = str(Path(__file__).resolve().parents[0]  / "test_models" / "logreg_iris_infertype.onnx")
model_path = str(Path(__file__).resolve().parents[1] / "examples" / "models")
data_path = str(Path(__file__).resolve().parents[1] / "examples" / "data")

@pytest.fixture
def iris_data():
    iris = pd.read_csv(f"{data_path}/iris_data.csv")
    return iris

def test_inp_error(iris_data):
    with pytest.raises(ValueError):
        model = estimate_treatment_effect(Y = iris_data['Y'], Z = iris_data['Z'], D = iris_data['D'])

def test_iris_estimation(iris_data):
    qps = estimate_qps_onnx(f"{model_path}/iris_logreg.onnx", data = iris_data[["X1", "X2", "X3", "X4"]])
    model = estimate_treatment_effect(qps, Y = iris_data['Y'], Z = iris_data['Z'], D = iris_data['D'])

    # Test second stage values
    ci = model.conf_int()
    stderr = model.std_errors
    coef = model.params
    fitted = model.fitted_values
    p = model.pvalues
    t = model.tstats
    r2 = model.rsquared
    resids = model.resids
    n = model.nobs

    obs_tokeep = np.nonzero((qps > 0) & (qps < 1))
    exog = sm.add_constant(qps[obs_tokeep])

    iv_check = IV2SLS(iris_data.loc[obs_tokeep[0], 'Y'], exog, iris_data.loc[obs_tokeep[0], 'D'], iris_data.loc[obs_tokeep[0], 'Z']).fit(cov_type='robust')
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
    assert np.allclose(coef.to_numpy().flatten(), validation_coef.to_numpy().flatten())
    assert n == validation_n
    assert np.allclose(resids.to_numpy().flatten(), validation_resids.to_numpy().flatten())
    assert np.allclose(fitted.to_numpy().flatten(), validation_fitted.to_numpy().flatten())
    assert np.allclose(stderr.to_numpy().flatten(), validation_stderr.to_numpy().flatten())
    assert np.allclose(t.to_numpy().flatten(), validation_t.to_numpy().flatten())
    assert np.allclose(r2, validation_r2)

    # Test first stage values
    fs = model.first_stage.individual['D']
    print(fs)
    ci = fs.conf_int()
    stderr = fs.std_errors
    coef = fs.params
    fitted = fs.fitted_values
    p = fs.pvalues
    t = fs.tstats
    r2 = fs.rsquared
    resids = fs.resids
    n = fs.nobs

    fs_check = iv_check.first_stage.individual['D']
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
    assert np.allclose(coef.to_numpy().flatten(), validation_coef.to_numpy().flatten())
    assert n == validation_n
    assert np.allclose(resids.to_numpy().flatten(), validation_resids.to_numpy().flatten())
    assert np.allclose(fitted.to_numpy().flatten(), validation_fitted.to_numpy().flatten())
    assert np.allclose(stderr.to_numpy().flatten(), validation_stderr.to_numpy().flatten())
    assert np.allclose(t.to_numpy().flatten(), validation_t.to_numpy().flatten())
    assert np.allclose(r2, validation_r2)

def test_iris_estimation_infer_positions(iris_data):
    qps = estimate_qps_onnx(f"{model_path}/iris_logreg.onnx", data = iris_data[["X1", "X2", "X3", "X4"]])
    iris_data['qps'] = qps
    model0 = estimate_treatment_effect(data = iris_data[['Y', 'Z', 'D', 'qps']])

    # Test second stage values
    ci0 = model0.conf_int()
    stderr0 = model0.std_errors
    coef0 = model0.params
    fitted0 = model0.fitted_values
    p0 = model0.pvalues
    t0 = model0.tstats
    r20 = model0.rsquared
    resids0 = model0.resids
    n0 = model0.nobs

    model1 = estimate_treatment_effect(data = iris_data[['Z', 'Y', 'D', 'qps']], Z_ind = 0)

    ci1 = model1.conf_int()
    stderr1 = model1.std_errors
    coef1 = model1.params
    fitted1 = model1.fitted_values
    p1 = model1.pvalues
    t1 = model1.tstats
    r21 = model1.rsquared
    resids1 = model1.resids
    n1 = model1.nobs

    assert np.allclose(ci0.to_numpy().flatten(), ci1.to_numpy().flatten())
    assert n0 == n1
    assert np.allclose(resids0.to_numpy().flatten(), resids1.to_numpy().flatten())
    assert np.allclose(fitted0.to_numpy().flatten(), fitted1.to_numpy().flatten())
    assert np.allclose(stderr0.to_numpy().flatten(), stderr1.to_numpy().flatten())
    assert np.allclose(t0.to_numpy().flatten(), t1.to_numpy().flatten())
    assert np.allclose(r20, r21)

    model2 = estimate_treatment_effect(data = iris_data[['Y', 'Z', 'D', 'qps']], qps = iris_data['qps'], Y_ind = 0, D_ind = 2)

    ci2 = model2.conf_int()
    stderr2 = model2.std_errors
    coef2 = model2.params
    fitted2 = model2.fitted_values
    p2 = model2.pvalues
    t2 = model2.tstats
    r22 = model2.rsquared
    resids2 = model2.resids
    n2 = model2.nobs

    assert np.allclose(ci1.to_numpy().flatten(), ci2.to_numpy().flatten())
    assert n1 == n2
    assert np.allclose(resids1.to_numpy().flatten(), resids2.to_numpy().flatten())
    assert np.allclose(fitted1.to_numpy().flatten(), fitted2.to_numpy().flatten())
    assert np.allclose(stderr1.to_numpy().flatten(), stderr2.to_numpy().flatten())
    assert np.allclose(t1.to_numpy().flatten(), t2.to_numpy().flatten())
    assert np.allclose(r21, r22)

    model3 = estimate_treatment_effect(data = iris_data[['Y', 'Z', 'D', 'qps']], qps = iris_data['qps'], Y_ind = 0, D_ind = 2, Z_ind = 1)

    ci3 = model3.conf_int()
    stderr3 = model3.std_errors
    coef3 = model3.params
    fitted3 = model3.fitted_values
    p3 = model3.pvalues
    t3 = model3.tstats
    r33 = model3.rsquared
    resids3 = model3.resids
    n3 = model3.nobs

    assert np.allclose(ci2.to_numpy().flatten(), ci3.to_numpy().flatten())
    assert n2 == n3
    assert np.allclose(resids2.to_numpy().flatten(), resids3.to_numpy().flatten())
    assert np.allclose(fitted2.to_numpy().flatten(), fitted3.to_numpy().flatten())
    assert np.allclose(stderr2.to_numpy().flatten(), stderr3.to_numpy().flatten())
    assert np.allclose(t2.to_numpy().flatten(), t3.to_numpy().flatten())
    assert np.allclose(r22, r33)

def test_counterfactual_estimation(iris_data):
    qps = estimate_qps_onnx(f"{sklearn_logreg}", data = iris_data[["X1", "X2", "X3", "X4"]], types=(np.float32,None))

    og_sess = rt.InferenceSession(sklearn_logreg)
    og_input = og_sess.get_inputs()[0].name
    og_preds = og_sess.run(['output_probability'], {og_input: iris_data[["X1", "X2", "X3", "X4"]].to_numpy().astype(np.float32)})[0]
    og_out = np.array([d[1] for d in og_preds])

    new_sess = rt.InferenceSession(sklearn_logreg_infer)
    new_input = new_sess.get_inputs()[0].name
    new_preds = new_sess.run(['output_probability'], {new_input: iris_data[["X1", "X2", "X3", "X4"]].to_numpy().astype(np.float32)})[0]
    new_out = np.array([d[1] for d in new_preds])

    cf_values, ols_results = estimate_counterfactual_ml(Y = iris_data.Y, Z = iris_data.Z, qps = qps, recs = og_out, cf_recs = new_out)

    # Check fitted OLS parameters
    ci = ols_results.conf_int()
    stderr = ols_results.std_errors
    coef = ols_results.params
    fitted = ols_results.fitted_values
    p = ols_results.pvalues
    t = ols_results.tstats
    r2 = ols_results.rsquared
    resids = ols_results.resids
    n = ols_results.nobs

    iris_data['qps'] = qps
    exog = sm.add_constant(iris_data[['Z', 'qps']])
    check_ols = IV2SLS(iris_data.Y, exog, None, None).fit(cov_type='unadjusted')

    ci_check = check_ols.conf_int()
    stderr_check = check_ols.std_errors
    coef_check = check_ols.params
    fitted_check = check_ols.fitted_values
    p_check = check_ols.pvalues
    t_check = check_ols.tstats
    r2_check = check_ols.rsquared
    resids_check = check_ols.resids
    n_check = check_ols.nobs

    assert np.allclose(ci.to_numpy().flatten(), ci_check.to_numpy().flatten())
    assert n == n_check
    assert np.allclose(resids.to_numpy().flatten(), resids_check.to_numpy().flatten())
    assert np.allclose(fitted.to_numpy().flatten(), fitted_check.to_numpy().flatten())
    assert np.allclose(stderr.to_numpy().flatten(), stderr_check.to_numpy().flatten())
    assert np.allclose(t.to_numpy().flatten(), t_check.to_numpy().flatten())
    assert np.allclose(r2, r2_check)

    check_cf_values = iris_data.Y.to_numpy() + coef_check['Z'] * (new_out - og_out)
    assert np.allclose(cf_values, check_cf_values)

def test_counterfactual_infer_positions(iris_data):
    qps = estimate_qps_onnx(f"{model_path}/iris_logreg.onnx", data = iris_data[["X1", "X2", "X3", "X4"]])
    iris_data['qps'] = qps

    og_sess = rt.InferenceSession(sklearn_logreg)
    og_input = og_sess.get_inputs()[0].name
    og_preds = og_sess.run(['output_probability'], {og_input: iris_data[["X1", "X2", "X3", "X4"]].to_numpy().astype(np.float32)})[0]
    og_out = np.array([d[1] for d in og_preds])

    new_sess = rt.InferenceSession(sklearn_logreg_infer)
    new_input = new_sess.get_inputs()[0].name
    new_preds = new_sess.run(['output_probability'], {new_input: iris_data[["X1", "X2", "X3", "X4"]].to_numpy().astype(np.float32)})[0]
    new_out = np.array([d[1] for d in new_preds])

    iris_data['recs'] = og_out
    iris_data['cf_recs'] = new_out
    cf_values0, model0 = estimate_counterfactual_ml(data = iris_data[['Y', 'Z', 'qps', 'recs', 'cf_recs']])

    # OLS values
    ci0 = model0.conf_int()
    stderr0 = model0.std_errors
    coef0 = model0.params
    fitted0 = model0.fitted_values
    p0 = model0.pvalues
    t0 = model0.tstats
    r20 = model0.rsquared
    resids0 = model0.resids
    n0 = model0.nobs

    cf_values1, model1 = estimate_counterfactual_ml(data = iris_data[['Y', 'Z', 'qps', 'recs', 'cf_recs']], Z_ind = 1)

    ci1 = model1.conf_int()
    stderr1 = model1.std_errors
    coef1 = model1.params
    fitted1 = model1.fitted_values
    p1 = model1.pvalues
    t1 = model1.tstats
    r21 = model1.rsquared
    resids1 = model1.resids
    n1 = model1.nobs

    assert np.allclose(ci0.to_numpy().flatten(), ci1.to_numpy().flatten())
    assert n0 == n1
    assert np.allclose(resids0.to_numpy().flatten(), resids1.to_numpy().flatten())
    assert np.allclose(fitted0.to_numpy().flatten(), fitted1.to_numpy().flatten())
    assert np.allclose(stderr0.to_numpy().flatten(), stderr1.to_numpy().flatten())
    assert np.allclose(t0.to_numpy().flatten(), t1.to_numpy().flatten())
    assert np.allclose(r20, r21)
    assert np.allclose(cf_values0, cf_values1)

    cf_values2, model2 = estimate_counterfactual_ml(data = iris_data[['Y', 'Z', 'qps', 'recs', 'cf_recs']], cf_recs = iris_data['cf_recs'], Y_ind = 0, qps_ind = 2)

    ci2 = model2.conf_int()
    stderr2 = model2.std_errors
    coef2 = model2.params
    fitted2 = model2.fitted_values
    p2 = model2.pvalues
    t2 = model2.tstats
    r22 = model2.rsquared
    resids2 = model2.resids
    n2 = model2.nobs

    assert np.allclose(ci1.to_numpy().flatten(), ci2.to_numpy().flatten())
    assert n1 == n2
    assert np.allclose(resids1.to_numpy().flatten(), resids2.to_numpy().flatten())
    assert np.allclose(fitted1.to_numpy().flatten(), fitted2.to_numpy().flatten())
    assert np.allclose(stderr1.to_numpy().flatten(), stderr2.to_numpy().flatten())
    assert np.allclose(t1.to_numpy().flatten(), t2.to_numpy().flatten())
    assert np.allclose(r21, r22)
    assert np.allclose(cf_values1, cf_values2)

    cf_values3, model3 = estimate_counterfactual_ml(data = iris_data[['Y', 'Z', 'qps', 'recs', 'cf_recs']], qps = iris_data['qps'], Y_ind = 0, qps_ind = 2, Z_ind = 1, recs = iris_data['recs'], cf_recs_ind = 4)

    ci3 = model3.conf_int()
    stderr3 = model3.std_errors
    coef3 = model3.params
    fitted3 = model3.fitted_values
    p3 = model3.pvalues
    t3 = model3.tstats
    r33 = model3.rsquared
    resids3 = model3.resids
    n3 = model3.nobs

    assert np.allclose(ci2.to_numpy().flatten(), ci3.to_numpy().flatten())
    assert n2 == n3
    assert np.allclose(resids2.to_numpy().flatten(), resids3.to_numpy().flatten())
    assert np.allclose(fitted2.to_numpy().flatten(), fitted3.to_numpy().flatten())
    assert np.allclose(stderr2.to_numpy().flatten(), stderr3.to_numpy().flatten())
    assert np.allclose(t2.to_numpy().flatten(), t3.to_numpy().flatten())
    assert np.allclose(r22, r33)
    assert np.allclose(cf_values2, cf_values3)
