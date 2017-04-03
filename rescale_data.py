import pandas as pd


def main():
    data = pd.read_csv("large_data/training.txt", sep='\t', encoding='ISO-8859-1', header=0)
    data.iloc[0:200, :].to_csv('data/training-small.csv')


if __name__ == '__main__':
    main()