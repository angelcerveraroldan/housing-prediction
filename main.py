import math

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import algos.neighbors

DATA_PATH = '/home/angelcr/programming/housing-prediction/data/housing-data.csv'


def data_accuracy(predictions, real):
    p, r = predictions, real

    differences = list(map(lambda x: abs(x[0] - x[1]), zip(p, r)))

    percentiles = np.percentile(differences, [5, 95])
    differences_filter = []
    for diff in differences:
        if percentiles[0] < diff < percentiles[1]:
            differences_filter.append(diff)

    print(f"Differences excluding outliers: {np.average(differences_filter)}")
    print(f"Differences: {np.average(differences)}")


def data_accuracy_hen(predictions, real): # as lists
    p, r = predictions, real
    w = 1.5 # Deviations
    array = np.stack((p, r), axis=1)
    difference = np.abs(array[:,0]-array[:,1])
    percentages = (difference/np.abs(array[:,1]))*100
    sorted_per = percentages[percentages[:].argsort()]
    median = sorted_per[int(len(percentages)/2)]

    # removes top and bottom 5% respectively
    variance = ((percentages - np.average(percentages)) ** 2 ).sum()/len(percentages)
    deviation = math.sqrt(variance)
    outlier_bool = np.logical_and(percentages < (np.average(percentages) + w * deviation), percentages > (np.average(percentages) - w * deviation))
    outliers = percentages[outlier_bool]

    print(f"average of 90%: {np.average(outliers)}")
    print(f"Median is: {median}")



def open_data():
    file = pd.read_csv(DATA_PATH)
    file.columns = ['lat', 'lon', 'date', 'price', 'rooms']

    # Convert date column from string to datetime
    file.date = pd.to_datetime(file.date, dayfirst=True)

    return file


def split_data(file: pd.DataFrame):
    return train_test_split(
        file.drop(columns=['price']), # Data excluding the price column
        file.price,                   # Price column
        test_size=0.2,                # Portion of data that will be used for test
        random_state=1,
    )


def main():
    # Get data ready for algorithm
    file = open_data()
    X_train, X_test, y_train, y_test = split_data(file)
    print(len(X_test))
    predictions = algos.neighbors.neighbors(100, X_train, y_train, X_test)

    data_accuracy_hen(predictions, y_test.values)

if __name__ == '__main__':
    main()

