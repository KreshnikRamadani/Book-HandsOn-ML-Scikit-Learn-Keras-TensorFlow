import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32

# let’s load the data using pandas
def load_housing_data(housing_path="datasets"):
    csv_path = os.path.join(housing_path,'housing', "housing.csv")
    return pd.read_csv(csv_path, thousands=',')

housing_data = load_housing_data()
print(housing_data.head())

# get a quick description of the data: total number of rows, attribute’s type, number of nonnull values
print(housing_data.info())

# There are 10 attributes = 9 Numeric + 1 categorical 
# what categories exist and how many districts are there
print(housing_data["ocean_proximity"].value_counts())

# get summary of each numerical attribute
print(housing_data.describe())

# plot a histogram for each numerical attribute
housing_data.hist(bins=50, figsize=(20,15))
plt.show()

# METHOD 1: split the data into training set and testing set
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shulffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shulffled_indices[:test_set_size]
    train_indices = shulffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#METHOD 2: split the data into training set and testing set
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio *2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda x: test_set_check(x, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

train_set, test_set = split_train_test(housing_data, 0.2)

print('Train set: ', len(train_set) )
print('Test set: ', test_set.head())
