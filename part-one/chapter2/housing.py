import os
import pandas as pd
import matplotlib.pyplot as plt

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
