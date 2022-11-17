from sklearn.neighbors import KNeighborsRegressor
import pandas as pd


def nearest_neighbors_regressor(n: int, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame):
    prices = y_train.values

    nbrs = KNeighborsRegressor(n_neighbors=n, algorithm='kd_tree')

    nbrs.fit(
        X_train[['lat', 'lon', 'rooms']].values,
        prices
    )

    return nbrs.predict(X_test[['lat', 'lon', 'rooms']].values)


def nearest_neighbors_regressor_log(n: int, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame):
    nbrs = KNeighborsRegressor(n_neighbors=n, algorithm='kd_tree')

    nbrs.fit(
        X_train[['lat', 'lon', 'rooms']].values,
        y_train.values
    )

    return nbrs.predict(X_test[['lat', 'lon', 'rooms']].values)
