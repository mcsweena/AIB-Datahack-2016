import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def data_import():
    data = pd.read_csv("large_data/training.txt", sep='\t', encoding='ISO-8859-1', header=0, low_memory=False)
    return data

def feature_selection(data):
    train_df = data[['County', 'Type']]
    test_df = data['EnergyRatingCat']

    # There is one empty value that needs to be filled
    test_df.fillna('Nan', inplace=True)
    return train_df, test_df


def factorise(train_df):
    train_df['County'] = pd.factorize(train_df.loc[:,'County'])[0]
    train_df['Type'] = pd.factorize(train_df.loc[:,'Type'])[0]
    return train_df


def cross_val(train_df, test_df):
    X_train, X_test, y_train, y_test = train_test_split(train_df, test_df,
                                                        test_size = 0.3,
                                                        random_state = 0)
    return X_train, X_test, y_train, y_test


def Logreg(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    print("Logistic Regression")
    print(accuracy_score(y_test, y_pred))


def main():
    data = data_import()
    train_df, test_df = feature_selection(data)
    train_df = factorise(train_df)
    X_train, X_test, y_train, y_test = cross_val(train_df, test_df)
    Logreg(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
