# AIB DataHack

### Encoding Issue

Encoding required for when importing the data. I get the following warning
when encoding isn't specified:

UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf8 in position 21: invalid start byte

Adding encoding='ISO-8859-1' to pd.read_csv() seems to solve this problem.


# Feature Selection

### Initial Features

Started off with two features; County and Type to initially set up the model
and ensure all components work. With these two features, Log Regression and
Random Forest algorithms were used. Random Forest gave a more accurate result
with over 20% accuracy.

### Dimensionality Features

Next, I took the dimensionality traits of the house as featrues to use;
GroundFloorArea, AvgWallU, AvgRoofU, AvgFloorU, AvgWindowU, AvgDoorU.

I first ran a small test to look at the number of null cells in each column.
AvgWallU stood out immediately here with over 46000 empty cells. This
will not be useful for prediction so I will remove it as an option.

