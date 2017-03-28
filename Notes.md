# AIB DataHack

Encoding required for when importing the data. I get the following warning
when encoding isn't specified:

UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf8 in position 21: invalid start byte

Adding encoding='ISO-8859-1' to pd.read_csv() seems to solve this problem.
