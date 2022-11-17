import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import algos.neighbor_regressor
import algos.kmeans

# Path to the training data
DATA_PATH = '/home/angelcr/programming/housing-prediction/data/housing-data.csv'


def data_accuracy(predictions, real):
    """
    Check the accuracy of the estimated prices
    """
    # This will be a list, the ith element of this list will be abs(prediction[i] - real[i])/real[i]
    differences = list(map(lambda x: abs(x[0] - x[1]) / x[1], zip(predictions, real)))

    # Find the value for the bottom t percentile and the top t percentile
    f = 0
    t = 90
    percentiles = np.percentile(differences, [f, t])
    differences_filter = []
    for diff in differences:
        # Keep only values in between f and t percentile
        if percentiles[0] < diff < percentiles[1]:
            differences_filter.append(diff)

    print(f"Differences excluding outliers: {np.average(differences_filter)}")


def open_data(path, t=False):
    """
    open the csv file and assign header names
    """
    file = pd.read_csv(path, header=None)

    if t:
        file.columns = ['lat', 'lon', 'date', 'rooms']
    else:
        file.columns = ['lat', 'lon', 'date', 'price', 'rooms']

    # Convert date column from string to datetime
    file.date = pd.to_datetime(file.date, dayfirst=True)

    # Convert dates into the year minus 2011
    dates = []
    for _, row in file.iterrows():
        dates.append(abs(row.date.year - 2013))

    file.date = dates

    return file


def split_data(file: pd.DataFrame):
    """
    Split the data into training and testing
    """
    return train_test_split(
        file.drop(columns=['price']),  # Data excluding the price column
        file.price,  # Price column
        test_size=0.2,  # Portion of data that will be used for test
        random_state=1,
    )


def main():
    # Get data ready for algorithm
    file = open_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(file)

    # predictions = algos.neighbor_regressor.nearest_neighbors_regressor_log(10, X_train, y_train, X_test)
    predictions = algos.kmeans.main(X_train, X_test, y_train)

    data_accuracy(predictions, y_test.values)


def predict():
    file = open_data(DATA_PATH)
    n_f = open_data('/home/angelcr/programming/housing-prediction/data/house-test.csv', True)

    X, y = file.drop(columns=['price']), file['price'].values
    X.info()
    n_f.info()
    predictions = algos.kmeans.main(X, n_f, y)
    print(f"Number of prediction: {len(predictions)}")
    ans = pd.read_csv('/home/angelcr/programming/housing-prediction/data/house-test.csv', header=None)
    ans.columns = ['lat', 'lon', 'date', 'rooms']
    ans['price'] = predictions

    ans.info()

    ans.to_csv('answ.csv', index_label='index')


if __name__ == '__main__':
    predict()
