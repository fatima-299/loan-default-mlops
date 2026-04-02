import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # Drop useless column
    df = df.drop(columns=["customer_id"])
    return df


def split_data(df):
    X = df.drop("default", axis=1)
    y = df["default"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = load_data("data/raw/Loan_Data.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print(X_train.shape, X_test.shape)
