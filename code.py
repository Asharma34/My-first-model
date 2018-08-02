import pandas as pd

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print(data.columns)

y = data.SalePrice

predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = data[predictors]

from sklearn.tree import DecisionTreeRegressor

# Define model
my_model = DecisionTreeRegressor()
my_model.fit(X, y)

print("Making predictions:")
print(X.head())
print("The predictions are")
print(my_model.predict(X.head()))
