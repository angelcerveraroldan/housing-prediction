from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as  np

def nearest_neighbors(n: int, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame):
    prices = y_train.values

    nbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree').fit(X_train[['lat', 'lon']].values)
    indexes_arrays = nbrs.kneighbors(X_test[['lat', 'lon']].values, 100, return_distance=False)

    predictions = []
    for indexes in indexes_arrays:
        predictions.append(
            list(map(lambda i: prices[i], indexes))
        )

    return list(map(lambda l: np.average(l), predictions))

def neighbors(n: int, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame):
    return nearest_neighbors(n, X_train, y_train, X_test)