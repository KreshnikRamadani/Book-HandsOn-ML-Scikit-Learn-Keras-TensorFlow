# Stand alone: End-to-End Machine Learning Project
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from utils.combined_attributes_adder import CombinedAttributesAdder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


''' let’s load the data using pandas '''
def load_housing_data(housing_path="datasets"):
    csv_path = os.path.join(housing_path,'housing', "housing.csv")
    return pd.read_csv(csv_path, thousands=',')

housing_data = load_housing_data()
print(housing_data.head())


''' Since the median_income is a continuous numerical attribute, you first need to create an income category attribute '''
housing_data["income_cat"] = pd.cut(housing_data["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1,2,3,4,5])

''' (Decomment the following line of code if you want to see) '''
#housing_data["income_cat"].hist() 

split_data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split_data.split(housing_data, housing_data["income_cat"]):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]

housing_data = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

''' transformation pipelines for the numerical attributes '''
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

''' Execute only the numeric pipeline:
    housing_num_tr = num_pipeline.fit_transform(housing_num) '''

''' create a copy of the data without the text attribute ocean_proximity, and fit the imputer instance to the training data'''
housing_num = housing_data.drop("ocean_proximity", axis=1)

''' FULL PIPELINE: transformation pipelines to clean up and prepare data (handling all columns) for ML algorithms '''
num_attribs = list(housing_num)     # we get the list of numerical column names
cat_attribs = ["ocean_proximity"]   # we get the list of categorical column names

full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing_data)
print('Full Pipeline: ', housing_prepared)

''' train a Linear Regression model '''
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

''' try it out on a few instances from the training set '''
some_data = housing_data.iloc[:5]
some_labels = housing_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))

''' Let’s measure this regression model’s RMSE on the whole training set '''
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Root Mean Square Error (RMSE) of LinearRegression: ", lin_rmse)


''' let’s try a more complex model to see how it does: DecisionTreeRegressor '''
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

''' let’s predict and evaluate it on the training set '''
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Root Mean Square Error (RMSE) of DecisionTreeRegressor: ", tree_rmse)
