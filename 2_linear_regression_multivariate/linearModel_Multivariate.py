import pandas as pd
import numpy as np
from sklearn import linear_model
import math

# price = m1*area + m2*bedrooms + m3*age + b
# here area, bedrroms, age are the independent variables (features)
# price is the dependent variable
# m1, m2, m3 are the coefficients
# b is intercept

# y = m1*x1 + m2*x2 + m3*x3 + b

#create dataframe
df = pd.read_csv("homeprices.csv")

# calculate median for filling the blank data
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

# create model
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

# Predict for 3000 sq ft area, 3 bedrooms and 40 year old home
reg = reg.predict([[3000,3,40]])

# check coefficient
print(reg.coef_)

# check intercept value
print(reg.intercept_)