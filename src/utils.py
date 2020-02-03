"""

 utils.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import pandas as pd

# Read csv file for (X, y, n_classes) data
def read_csv(csv_filename, target_name="y", normalize=False):
    df = pd.read_csv(csv_filename, delimiter=",", dtype={target_name: str})
    if list(df.columns.values).count(target_name) != 1: # check target
        raise Exception("Need exactly 1 count of '{}' in {}".format(target_name, csv_filename))
    target2idx = {target: idx for idx, target in enumerate(sorted(list(set(df[target_name].values))))}
    X = df.drop([target_name], axis=1).values
    y = np.vectorize(lambda x: target2idx[x])(df[target_name].values)
    n_classes = len(target2idx.keys())
    if X.shape[0] != y.shape[0]:
        raise Exception("X.shape = {} and y.shape = {} are inconsistent!".format(X.shape, y.shape))
    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y, n_classes

# Randomly permute [0,N] and extract indices for each fold
def crossval_folds(N, n_folds, seed=1):
    np.random.seed(seed)
    idx_all_permute = np.random.permutation(N)
    N_fold = int(N/n_folds)
    idx_folds = []
    for i in range(n_folds):
        start = i*N_fold
        end = min([(i+1)*N_fold, N])
        idx_folds.append(idx_all_permute[start:end])
    return idx_folds