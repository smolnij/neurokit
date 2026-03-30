import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_dataset(parameters, data="../data/raw/abalone.data"):
    """Docstring: briefly describe what the function does."""

    #TODO DESCRIBE FUNCTION
    x = pd.read_csv(data)

    #TODO DEFINE UNSUPPORTED DATA TYPES (STRINGS) AND CONVERT TO SUPPORTED
    #TODO RIGHT NOW FUNCTION ASSUMES THAT ANSWERS ARE IN THE LAST COLUMN, NEED TO MAKE A PARAMETER FOR THIS
    #TODO REMOVE DEBUG INFO, FORMAT HELPFUL INFO
    print("INITIAL: " , x.head())
    print(x.shape)  # rows, columns
    # print(x.columns)    # column names
    # print(x.info())     # data types
    # print(x.describe()) # statistics

    print(x.dtypes)

    # x = features, y = labels

    # iloc → position-based indexing, : → all rows, -1 → last column
    y = x.iloc[:, -1]

    #All, but last
    x = x.iloc[:, :-1]
    print("ANSWERS CUT OFF: " , x.head())

    x = pd.get_dummies(x)

    # First split: train+val and test
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=42
    )

    # second split: validation vs test
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42
    )


#TODO DOCUMENT SCALING
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    return x_train, y_train, x_test, y_test, x_val, y_val
