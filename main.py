import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score



def data_import():
    data = pd.read_csv("large_data/training.txt", sep='\t', encoding='ISO-8859-1', header=0, low_memory=False)
    test = pd.read_csv("large_data/testing.txt", sep='\t', encoding='ISO-8859-1', header=0, low_memory=False)
    # data = pd.read_csv('data/training-small.csv')

    # Print out column names one by one
    # columns = data.columns
    # for i in columns:
    #     print(i, type(i))
    print("Data imported...")
    return data, test

def feature_selection(data, test):
    train_df = data.loc[:,['County', 'Type', 'AvgRoofU', 'AvgFloorU', 'AvgWindowU', 'AvgDoorU']]
    label_df = data.loc[:,'EnergyRatingCat']

    test_df = test.loc[:, ['County', 'Type', 'AvgRoofU', 'AvgFloorU', 'AvgWindowU', 'AvgDoorU']]
    labels_test = test.loc[:, 'EnergyRatingCat']


    # Reduce Categories to just Letters
    label_df = label_df.str[0:1]
    labels_test = labels_test.str[0:1]

    # Characters removed from each cell and remainder converted to float
    #train_df['GroundFloorArea'] = train_df['GroundFloorArea'].str[:-5].astype(float)

    # print("No. of Empty Columns: ", train_df.isnull().sum())
    # Count Number of empty cells in each column

    # Fill Empty Columns
    print("\nfilling columns...")
    train_df['County'].fillna('Nan', inplace=True)
    print('County nulls filled')
    train_df['Type'].fillna('Nan', inplace=True)
    print('Type nulls filled')

    # train_df.fillna(train_df.mean(), inplace=True)

    train_df['AvgRoofU'].fillna(train_df['AvgRoofU'].mean(), inplace=True)
    train_df['AvgFloorU'].fillna(train_df['AvgFloorU'].mean(), inplace=True)
    train_df['AvgWindowU'].fillna(train_df['AvgWindowU'].mean(), inplace=True)
    train_df['AvgDoorU'].fillna(train_df['AvgDoorU'].mean(), inplace=True)

    print('Rest of the nulls filled')
    label_df.fillna('Nan', inplace=True)
    labels_test.fillna('Nan', inplace=True)
    print("Features isolated and empty values are filled...")

    return train_df, label_df, test_df, labels_test


def factorise(train_df):
    train_df['County'] = pd.factorize(train_df.loc[:,'County'])[0]
    train_df['Type'] = pd.factorize(train_df.loc[:,'Type'])[0]
    print("\nFeatures are factorised...")
    return train_df


def pre_processing(train_df):
    print(train_df.columns)
    scaler = MinMaxScaler()

    train_df.loc[:, 'AvgRoofU'] = scaler.fit_transform(train_df.loc[:, 'AvgRoofU'])
    train_df.loc[:, 'AvgFloorU'] = scaler.fit_transform(train_df.loc[:, 'AvgFloorU'])
    train_df.loc[:, 'AvgWindowU'] = scaler.fit_transform(train_df.loc[:, 'AvgWindowU'])
    train_df.loc[:, 'AvgDoorU'] = scaler.fit_transform(train_df.loc[:, 'AvgDoorU'])

    print("\nPreprocessing complete...")
    return train_df


def cross_val(train_df, label_df):
    X_train, X_test, y_train, y_test = train_test_split(train_df, label_df,
                                                        test_size = 0.3,
                                                        random_state = 0)
    print("\nCross validation complete...")
    return X_train, X_test, y_train, y_test


def PrincCompAnalysis(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_reduced = pca.transform(X)
    print("Reduced dataset shape:", X_reduced.shape)
    print("Explained Variance: ", pca.explained_variance_)
    print("Components: ", pca.components_)

    # return train_df


def Logreg(X_train, X_test, y_train, y_test, test_df, labels_test):
    logreg = LogisticRegression()

    scores = cross_val_score(logreg, X_train, y_train, cv=5)

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(test_df)
    print("\nLogistic Regression")
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Test data accuracy score: ", accuracy_score(labels_test, y_pred))


def RandomForest(X_train, X_test, y_train, y_test):
    RF = RandomForestClassifier(n_estimators=100)
    RF.fit(X_train, y_train)

    y_pred = RF.predict(X_test)
    print("\nRandom Forest")
    print(accuracy_score(y_test, y_pred))


def main():
    data, test = data_import()
    train_df, label_df, test_df, labels_test = feature_selection(data, test)
    train_df = factorise(train_df)
    train_df = pre_processing(train_df)
    # train_df = PrincCompAnalysis(train_df)

    X_train, X_test, y_train, y_test = cross_val(train_df, label_df)


    # Algorithms
    Logreg(X_train, X_test, y_train, y_test, test_df, labels_test)
    # RandomForest(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
