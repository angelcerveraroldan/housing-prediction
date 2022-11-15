import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = '/home/angelcr/programming/housing-prediction/data/housing-data.csv'


def open_data():
    file = pd.read_csv(DATA_PATH)
    file.columns = ['lat', 'lon', 'date', 'price', 'rooms']

    # Convert date column from string to datetime
    file.date = pd.to_datetime(file.date)

    return file


def split_data(file: pd.DataFrame):
    return train_test_split(
        file.drop(columns=['price']), # Data excluding the price column
        file.price,                   # Price column
        test_size=0.1                 # Portion of data that will be used for test
    )


def main():
    # Get data ready for algorithm
    file = open_data()
    X_train, X_test, y_train, y_test = split_data(file)


if __name__ == '__main__':
    main()

