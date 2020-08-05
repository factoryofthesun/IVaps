# Generate sample data
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
import onnxruntime as rt
from pathlib import Path

from mlisne.dataset import IVEstimatorDataset
from mlisne.helpers import estimate_qps


def generate_iris_data():
    """Simulate historical ML data using sklearn's logistic regression trained on iris
    Machine learning recommendation = Z
    Treatment assignment D: follows Z 75% of the time
    Y = 1 + N(5, 1)*D + N(1,1)*QPS + N(0,1)"""

    sklearn_logreg = str(Path(__file__).resolve().parents[1] / "models" / "iris_logreg.onnx")
    iris = load_iris()
    X = iris.data

    # Sample 10000 observations from original X data
    X_sample = np.apply_along_axis(np.random.choice, axis=0, arr=X, size=int(1e4)).astype(np.float32)
    sess = rt.InferenceSession(sklearn_logreg)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[1].name
    rec_dict_probs = sess.run([label_name], {input_name: X_sample})[0]
    rec_probs = [d[1] for d in rec_dict_probs]
    rec_draws = np.random.uniform(size=int(1e4))
    Z = (rec_draws <= rec_probs).astype(int)

    treat_probs = np.random.uniform(size=int(1e4))
    D = []
    for i in range(len(treat_probs)):
        if treat_probs[i] >= 0.75:
            if Z[i] == 1:
                D.append(0)
            else:
                D.append(1)
        else:
            D.append(Z[i])
    D = np.array(D)
    qps = estimate_qps(X_sample, S=100, delta=0.8, ML_onnx=sklearn_logreg)
    Y = 1 + np.random.normal(5,1) * D + np.random.normal(1,1) * qps + np.random.normal(0,2,size=int(1e4))

    data = np.concatenate((Y[:,np.newaxis], Z[:,np.newaxis], D[:,np.newaxis], X_sample, qps[:,np.newaxis]), axis=1)
    x_cols = [f'X_{i}' for i in range(X.shape[1])]
    pd_cols = ['Y', 'Z', 'D'] + x_cols + ['QPS']
    pd_dat = pd.DataFrame(data=data, columns=pd_cols)
    pd_dat.to_csv("iris_data.csv", index=False)

if __name__ == "__main__":
    generate_iris_data()
