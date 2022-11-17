import math

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Number of clusters into which we will divide ireland
CLUSTER_NUM = 500


def cluster(df: pd.DataFrame):
    """
    Divide Ireland into clusters based on latitude and longitude using k-means algorithm

    :return:
        - an array containing the coordinates for the center of each cluster
    """
    cluster_on = df[['lat', 'lon']]

    kmeans = KMeans(
        init='random',
        n_clusters=CLUSTER_NUM,
        n_init=20,
        max_iter=900
    )

    kmeans.fit(cluster_on)
    df['cluster'] = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return centroids


def linear_regression(x, y):
    reg = LinearRegression()
    reg.fit(x, y)

    return reg


# Hot encode
def hot_encode(df):
    """
    Hot encode the clusters
    - We cannot put the cluster number into the regression model as it is categorical data
    """
    dummies = pd.get_dummies(df['cluster'], prefix='cluster', drop_first=True)
    return pd.concat([df, dummies], axis=1)


def assign_cluster(df, centroids):
    """
    Find the nearest cluster to each house
    """
    cluster_index = []
    for _, row in df.iterrows():
        x = row.lat
        y = row.lon

        # Find the distance from the point to each of the centroids, the nearest one will be the cluster it belongs to
        distances = list(map(lambda c: math.dist([x, y], c), centroids))
        closest_point = min(distances)

        # Cluster number
        index = distances.index(closest_point)

        cluster_index.append(index)

    return cluster_index


def main(X_train, X_test, y_train):
    cluster_on = ['lat', 'lon']

    # Change each price to log2
    # Eg, if the price of a house was 1000, it will be replaced with log_2(1000)
    # This makes linear regression more accurate
    y_train = list(map(lambda p: np.log2(p), y_train))

    # Assign each residence a cluster
    centroids = cluster(X_train)
    X_test['cluster'] = assign_cluster(X_test, centroids)

    # Hot encode the clusters
    X_train = hot_encode(X_train)
    X_test = hot_encode(X_test)

    # Drop the columns that will not be used in the regressor
    X_train = X_train.drop(columns=(cluster_on + ['cluster']))
    X_test = X_test.drop(columns=(cluster_on + ['cluster']))

    # Generate regressor
    reg = linear_regression(X_train, y_train)

    predictions = reg.predict(X_test)
    # Put the results back into price ( they are predicted in log_2 form)
    results = list(map(lambda p: 2 ** p, predictions))

    return results
