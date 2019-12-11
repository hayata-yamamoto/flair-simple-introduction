import pandas as pd

from sklearn.datasets import fetch_20newsgroups


def main() -> None: 
    x_train, y_train = fetch_20newsgroups(return_X_y=True)
    x_test, y_test = fetch_20newsgroups(return_X_y=True)

    df = pd.Datetime([x_train, y_train], columns=['text', 'y'])
    print(df.head())

if __name__ == '__main__':
    main()
