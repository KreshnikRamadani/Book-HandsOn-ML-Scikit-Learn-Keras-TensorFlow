# Stand alone: End-to-End Machine Learning Project
import os
import joblib
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


''' let’s load the data using pandas '''
def load_housing_data(housing_path="datasets"):
    csv_path = os.path.join(housing_path,'housing', "housing.csv")
    return pd.read_csv(csv_path, thousands=',')

housing_data = load_housing_data()
#print(housing_data.head())

''' Since the median_income is a continuous numerical attribute, you first need to create an income category attribute '''
housing_data["income_cat"] = pd.cut(housing_data["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1,2,3,4,5])

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

''' Execute only the numeric pipeline: housing_num_tr = num_pipeline.fit_transform(housing_num) '''

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
#print('Full Pipeline: ', housing_prepared)

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
print("\n\nLinearRegression - Root Mean Square Error (RMSE): ", lin_rmse)


''' let’s try a more complex model to see how it does: DecisionTreeRegressor '''
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

''' let’s predict and evaluate it on the training set '''
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("\n\nDecisionTreeRegressor - Root Mean Square Error (RMSE): ", tree_rmse)

''' K-fold cross-validation feature to compute the same scores for the Decision Tree Regressor  '''
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

print('Decision Tree Regressor model')
display_scores(tree_rmse_scores)

''' compute the same scores for the Linear Regression model '''
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

print("\n\nLinear Regression model")
display_scores(lin_rmse_scores)

''' let's try RandomForestRegressor model '''
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

''' let’s predict and evaluate it on the training set '''
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("\n\nRandomForestRegressor - Root Mean Square Error (RMSE): ", forest_rmse)

''' compute the same scores for the Random Forest Regression model '''
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("Random Forest Regression model ")
display_scores(forest_rmse_scores)

''' save every model we experiment '''
#joblib.dump(lin_reg, "lin_reg.pkl")
#joblib.dump(tree_reg, "tree_reg.pkl")
#joblib.dump(forest_reg, "forest_reg.pkl")

''' come back easily to any model '''
#lin_reg_loaded = joblib.load("lin_reg.pkl")
#tree_reg_loaded = joblib.load("tree_reg.pkl")
#forest_reg_loaded = joblib.load("forest_reg.pkl")


""" ''' FINE-TUNE the Model - best combination of hyperparameter values for the RandomForestRegressor '''
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

''' get the best combination of parameters '''
print('Best combination of parameters: ', grid_search.best_params_) # Result: {'max_features': 8, 'n_estimators': 30}

''' get the best estimator '''
print('Best estimator: ', grid_search.best_estimator_) # Result: {'max_features': 8, 'n_estimators': 30}

''' get the evaluation scores '''
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

 """