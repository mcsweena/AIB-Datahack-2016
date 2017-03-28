import pandas as pd
from sklearn import cross_validation


def data_import():
    data = pd.read_csv("large_data/training.txt", sep='\t', encoding='ISO-8859-1', header=0, low_memory=False)

    return data

def feature_selection():
    pass


def cross_val():
    pass


def Logreg():
    pass



def main():
    data = data_import()



if __name__ == '__main__':
    main()
