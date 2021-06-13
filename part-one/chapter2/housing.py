import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from utils.combined_attributes_adder import CombinedAttributesAdder

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


''' let’s load the data using pandas '''
def load_housing_data(housing_path="datasets"):
    csv_path = os.path.join(housing_path,'housing', "housing.csv")
    return pd.read_csv(csv_path, thousands=',')

housing_data = load_housing_data()
print(housing_data.head())

''' get a quick description of the data: total number of rows, attribute’s type, number of nonnull values '''
print(housing_data.info())

''' There are 10 attributes = 9 Numeric + 1 categorical 
    what categories exist and how many districts are there '''
print(housing_data["ocean_proximity"].value_counts())

''' get summary of each numerical attribute '''
print(housing_data.describe())

''' plot a histogram for each numerical attribute (Decomment the following line of code if you want to see)'''
#housing_data.hist(bins=50, figsize=(20,15))


''' METHOD 1: split the data into training set and testing set '''
""" def split_train_test(data, test_ratio):
    np.random.seed(42)
    shulffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shulffled_indices[:test_set_size]
    train_indices = shulffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] """
# train_set, test_set = split_train_test(housing_data, 0.2)
# print('Train set: ', len(train_set) )
# print('Test set: ', test_set.head())

''' METHOD 2: split the data into training set and testing set '''
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio *2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda x: test_set_check(x, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

'''Unfortunately, the housing dataset does not have an identifier column. 
ONE WAY: The simplest solution is to use the row index as the ID:'''
housing_data_with_id = housing_data.reset_index()
#train_set, test_set = split_train_test_by_id(housing_data_with_id,0.2,"index")

'''
But if the data new data gets appended to the end of the dataset and some (old) row data gets deleted
ANOTHER WAY - SOLUTION: try to use the most stable features to build a unique identifier, for example: id = longitude * 1000 + latitude '''
# housing_data_with_id["id"] = housing_data["longitude"] * 1000 + housing_data["latitude"]
# train_set, test_set = split_train_test_by_id(housing_data_with_id, 0.2, "id")

""" Scikit-Learn provides a few functions to split datasets into multiple subsets in various ways. 
    The simplest function is train_test_split() """
train_set, test_set = train_test_split(housing_data, test_size= 0.2, random_state= 42)


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

''' looking at the income category proportions in the test set '''
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

''' remove the income_cat attribute so the data is back to its original state '''
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

''' create a copy so that you can play with it without harming the training set: '''
housing_data = strat_train_set.copy()

''' Since there is geographical information (latitude and longitude), it is a good idea to create a scatterplot of all districts to visualize the data 
Setting the alpha option to 0.1 makes it much easier to visualize the places where there is a high density of data points'''
housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


''' let’s look at the housing prices '''
housing_data.plot(kind="scatter", 
                     x="longitude", 
                     y="latitude",
                     alpha=0.4,
                     s=housing_data["population"]/100,
                     label="population",
                     figsize=(10,7),
                     c="median_house_value",
                     cmap=plt.get_cmap("jet"), 
                     colorbar=True)
#plt.legend()

''' dataset is not too large: compute the standard correlation coefficient '''
corr_matrix = housing_data.corr()

''' let’s look at how much each attribute correlates with the median house value '''
print(corr_matrix["median_house_value"].sort_values(ascending=False))

''' plot every numerical attribute against every (let’s just focus on a few promising attributes) other numerical attribute.'''
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing_data[attributes], figsize=(12,8))

''' let’s zoom in on their (median_house_value is the median_income) correlation scatterplot '''
housing_data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

''' create these new attributes: number of rooms per household, number of bedrooms by itself (rooms), population per household '''
housing_data["rooms_per_household"] = housing_data["total_rooms"]/housing_data["households"]
housing_data["bedrooms_per_room"] = housing_data["total_bedrooms"]/housing_data["total_rooms"]
housing_data["population_per_household"] = housing_data["population"]/housing_data["households"]

''' and now let’s look at the correlation matrix again '''
corr_matrix = housing_data.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

''' (De)comment this line if you want to show/hide all plots '''
#plt.show()



''' DATA CLEANING '''
housing_data = strat_test_set.drop("median_house_value", axis=1)
housing_labels = strat_test_set["median_house_value"].copy()

print(" \nCount missing values: ", housing_data["total_bedrooms"].isnull().sum())

''' There are three option to fix missing values, Decomment/comment.
    1. Get rid of the corresponding districts. '''
#housing_data.dropna(subset=["total_bedrooms"])

''' 2. Get rid of the whole attribute. '''
#housing_data.drop(["total_bedrooms"])

''' 3. Set the values to some value (zero, the mean, the median, etc.). '''
#median = housing_data["total_bedrooms"].median()
#housing_data["total_bedrooms"].fillna(median,inplace=True)
#print(" \nCount missing values: ", housing_data["total_bedrooms"].isnull().sum())


''' Scikit-Learn provides a handy class to take care of missing values:
    1. create a SimpleImputer instance
    2. specify that you want to replace each attribute’s missing values with the median of that attribute: '''
imputer = SimpleImputer(strategy="median")

''' create a copy of the data without the text attribute ocean_proximity, and fit the imputer instance to the training data'''
housing_num = housing_data.drop("ocean_proximity", axis=1)
print(" \nCount missing values: ", housing_num["total_bedrooms"].isnull().sum())
imputer.fit(housing_num)

''' The imputer has simply computed the median of each attribute and stored the result in its statistics_ instance variable. Let's print...'''
print("statistics_: ",imputer.statistics_)

''' use this “trained” imputer to transform the training set by replacing missing values with the learned medians '''
X = imputer.transform(housing_num)

''' result is a plain NumPy array containing the transformed features. If you want to put it back into a pandas DataFrame '''
housing_tr = pd.DataFrame(X,columns=housing_num.columns, index=housing_num.index)

print(" \nCount missing values: ", housing_tr["total_bedrooms"].isnull().sum())

'''NEXT: Handling Text and Categorical Attributes 
    - create a separate DataFrame only with categories
    - Let’s convert these categories from text to numbers '''

ordinal_encoder = OrdinalEncoder()
housing_cat = housing_data.select_dtypes(include=['object']).copy()
print("categorical values: ", housing_cat.head(10))
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print("categories from text to numbers: ", housing_cat_encoded[:10])

''' get the list of categories (values) for each categorical attribute '''
print(ordinal_encoder.categories_)

''' convert categorical values into one-hot vectors '''
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot.toarray())



''' my own small transformer class that adds the combined attributes '''
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing_data.values)
print('Creating combined attributes: ', housing_extra_attribs)


''' transformation pipelines for the numerical attributes '''
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

''' Execute only the numeric pipeline:
    housing_num_tr = num_pipeline.fit_transform(housing_num) '''

''' FULL PIPELINE: transformation pipelines to clean up and prepare data (handling all columns) for ML algorithms '''
num_attribs = list(housing_num)     # we get the list of numerical column names
cat_attribs = ["ocean_proximity"]   # we get the list of categorical column names

full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing_data)
print('Full Pipeline: ', housing_prepared)
