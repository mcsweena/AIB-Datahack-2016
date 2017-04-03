import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


def data_import():
    data = pd.read_csv("large_data/training.txt", sep='\t', encoding='ISO-8859-1', header=0, low_memory=False)
    #data = pd.read_csv('data/training-small.csv')

    # Print out column names one by one
    # columns = data.columns
    # for i in columns:
    #     print(i, type(i))

    return data

def feature_selection(data):
    train_df = data.loc[:,['County', 'Type', 'GroundFloorArea', 'AvgRoofU', 'AvgFloorU',
                           'AvgWindowU', 'AvgDoorU']]
    test_df = data.loc[:,'EnergyRatingCat']

    # Characters removed from each cell and remainder converted to float
    train_df['GroundFloorArea'] = train_df['GroundFloorArea'].str[:-5].astype(float)

    # Count Number of empty cells in each column
    # print("No. of Empty Columns: ", train_df.isnull().sum())

    # Fill Empty Columns
    train_df.fillna(train_df.mean(), inplace=True)
    test_df.fillna('Nan', inplace=True)
    print("Features isolated and empty values are filled...")
    return train_df, test_df


def factorise(train_df):
    train_df['County'] = pd.factorize(train_df.loc[:,'County'])[0]
    train_df['Type'] = pd.factorize(train_df.loc[:,'Type'])[0]
    print("Features are factorised...")
    return train_df


def pre_processing(train_df):
    print(train_df.columns)
    scaler = MinMaxScaler()

    train_df.loc[:, 'GroundFloorArea'] = scaler.fit_transform(train_df.loc[:, 'GroundFloorArea'])
    train_df.loc[:, 'AvgRoofU'] = scaler.fit_transform(train_df.loc[:, 'AvgRoofU'])
    train_df.loc[:, 'AvgFloorU'] = scaler.fit_transform(train_df.loc[:, 'AvgFloorU'])
    train_df.loc[:, 'AvgWindowU'] = scaler.fit_transform(train_df.loc[:, 'AvgWindowU'])
    train_df.loc[:, 'AvgDoorU'] = scaler.fit_transform(train_df.loc[:, 'AvgDoorU'])

    print(train_df.head())


    print("Preprocessing complete...")
    return train_df


def cross_val(train_df, test_df):
    X_train, X_test, y_train, y_test = train_test_split(train_df, test_df,
                                                        test_size = 0.3,
                                                        random_state = 0)
    print("Cross validation complete...")
    return X_train, X_test, y_train, y_test


def Logreg(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    print("Logistic Regression")
    print(accuracy_score(y_test, y_pred))


def RandomForest(X_train, X_test, y_train, y_test):
    RF = RandomForestClassifier(n_estimators=50)
    RF.fit(X_train, y_train)

    y_pred = RF.predict(X_test)
    print("Random Forest")
    print(accuracy_score(y_test, y_pred))


def main():
    data = data_import()
    train_df, test_df = feature_selection(data)
    train_df = factorise(train_df)
    train_df = pre_processing(train_df)
    X_train, X_test, y_train, y_test = cross_val(train_df, test_df)
    Logreg(X_train, X_test, y_train, y_test)
    #RandomForest(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
