import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split

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

''' plot a histogram for each numerical attribute '''
housing_data.hist(bins=50, figsize=(20,15))
plt.show()

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

