import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model # Model-based learning algorithm
import sklearn.neighbors    # Instance-based learning algorithm
import os

# Load the data
datapath = os.path.join("datasets", "lifesat", "")
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")

#Print the head of oecd_bli_2015 dataset
#print(oecd_bli.head())

#Print the head of gdp_per_capita dataset
# print(gdp_per_capita.head())

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

#print(x)
#print(y)

# Select a linear model
model_based = sklearn.linear_model.LinearRegression()

instance_based  = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model
model_based.fit(x, y)
instance_based.fit(x, y)

X_new = [[22587]] # Cyprus's GDP per
print('Model-based prediction: ',model_based.predict(X_new))       # outputs [[ 5.96242338]]
print('Instance-based prediction: ', instance_based.predict(X_new))    # outputs [[ 5.77]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()