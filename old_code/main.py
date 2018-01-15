import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from time import time
#import matplotlib.pyplot as plt


def data_import():
    data = pd.read_csv("large_data/training.txt", sep='\t', encoding='ISO-8859-1', header=0, low_memory=False)
    # test = pd.read_csv("large_data/testing.txt", sep='\t', encoding='ISO-8859-1', header=0, low_memory=False)
    # data = pd.read_csv('data/training-small.csv')

    # Print out column names one by one
    # columns = data.columns
    # for i in columns:
    #     print(i, type(i))
    print("Data imported...")
    return data


def remove_null_columns(data):
    # Count Number of empty cells in each column
    org_num_cols = data.isnull().sum()
    columns_to_drop = []

    for i in data.columns:
        if data[i].isnull().sum() > 100:
            columns_to_drop.append(i)

    new_df = data.drop(columns_to_drop, axis=1)

    print("Original df len: ", len(org_num_cols))
    print("New df len: ", len(new_df.columns))

    # print("\nGrouped Column Types")
    # print(new_df.columns.to_series().groupby(new_df.dtypes).groups)

    cols_to_drop = []
    for i in new_df.columns:
        if new_df[i].dtype != 'float64':
            cols_to_drop.append(i)

    # Don't remove Labels Column
    if "EnergyRatingCat" in cols_to_drop:
        cols_to_drop.remove("EnergyRatingCat")

    cols_to_drop.append('EnergyRatingCont')
    num_cols = cols_to_drop

    new_df_float = new_df.drop(cols_to_drop, axis=1)
    print("New df len: ", len(new_df_float.columns))

    return new_df_float

def plot_nulls(data):
    null_dict = {}
    for i in data.columns:
        null_dict[i] = data[i].isnull().sum()

    plt.bar(range(len(null_dict)), null_dict.values(), align='center')
    plt.xticks(range(len(null_dict)), list(null_dict.keys()))

    plt.show()


def feature_selection(data):
    # train_df = data.loc[:,['County','Type','AvgRoofU', 'AvgFloorU', 'AvgWindowU', 'AvgDoorU']]
    train_df = data.drop(['EnergyRatingCat'], axis=1)
    label_df = data.loc[:,'EnergyRatingCat']

    # Reduce Categories to just Letters
    # label_df = label_df.str[0:1]

    # Characters removed from each cell and remainder converted to float
    #train_df['GroundFloorArea'] = train_df['GroundFloorArea'].str[:-5].astype(float)

    # Fill Empty Columns
    # train_df['County'].fillna('Nan', inplace=True)
    # train_df['Type'].fillna('Nan', inplace=True)

    train_df.fillna(train_df.mean(), inplace=True)

    # train_df['AvgRoofU'].fillna(train_df['AvgRoofU'].mean(), inplace=True)
    # train_df['AvgFloorU'].fillna(train_df['AvgFloorU'].mean(), inplace=True)
    # train_df['AvgWindowU'].fillna(train_df['AvgWindowU'].mean(), inplace=True)
    # train_df['AvgDoorU'].fillna(train_df['AvgDoorU'].mean(), inplace=True)

    print('Rest of the nulls filled')
    label_df.fillna('Nan', inplace=True)
    print("Features isolated and empty values are filled...")

    return train_df, label_df


def factorise(train_df):
    train_df['County'] = pd.factorize(train_df.loc[:,'County'])[0]
    train_df['Type'] = pd.factorize(train_df.loc[:,'Type'])[0]
    print("\nFeatures are factorised...")
    return train_df


def pre_processing(train_df):
    print(train_df.columns)
    scalar = MinMaxScaler()

    train_df = scalar.fit_transform(train_df)

    # train_df.loc[:, 'AvgRoofU'] = scaler.fit_transform(train_df.loc[:, 'AvgRoofU'])
    # train_df.loc[:, 'AvgFloorU'] = scaler.fit_transform(train_df.loc[:, 'AvgFloorU'])
    # train_df.loc[:, 'AvgWindowU'] = scaler.fit_transform(train_df.loc[:, 'AvgWindowU'])
    # train_df.loc[:, 'AvgDoorU'] = scaler.fit_transform(train_df.loc[:, 'AvgDoorU'])

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


def Logreg(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    print("\nLogistic Regression")
    print("Test data accuracy score: ", accuracy_score(y_test, y_pred))


def RandomForest(X_train, X_test, y_train, y_test):
    RF = RandomForestClassifier(n_estimators=100, min_samples_split=2)
    RF.fit(X_train, y_train)

    y_pred = RF.predict(X_test)
    print("\nRandom Forest")
    print(accuracy_score(y_test, y_pred))
    print(RF.feature_importances_)


def Adaboost(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== AdaBoost ===")
    print(accuracy_score(y_pred, y_test))


def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("\nK Nearest Neighbour")
    print(accuracy_score(y_test, y_pred))


def main():
    data = data_import()
    data = remove_null_columns(data)
    #plot_nulls(data)
    train_df, label_df = feature_selection(data)
    # train_df = factorise(train_df)
    train_df = pre_processing(train_df)
    # train_df = PrincCompAnalysis(train_df)

    X_train, X_test, y_train, y_test = cross_val(train_df, label_df)


    ### Algorithms ###

    ### Logistic Regression
    # Logreg(X_train, X_test, y_train, y_test)

    ### Random Forest
    # start = time()
    # RandomForest(X_train, X_test, y_train, y_test)
    # end = time()
    # print("Time: ", end - start)

    ### AdaBoost
    # Adaboost(X_train, X_test, y_train, y_test)

    ### K Nearest Neighbours
    # start = time()
    # KNN(X_train, X_test, y_train, y_test)
    # end = time()
    # print("Time: ", end - start)


if __name__ == '__main__':
    main()
