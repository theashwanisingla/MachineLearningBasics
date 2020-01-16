import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#Creating a dataframe from csv file
df = pd.read_csv('homeprices.csv')

# plotting the datapoints
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')

#creating new features 
new_df = df.drop('price',axis='columns')
price = df.price

# Create linear regression object
reg = linear_model.LinearRegression()

#fit data to the model for training 
reg = reg.fit(new_df,price)

#Predict the price for area of 3700
result = reg.predict([[3700]])
print(f"Predicted price is {int(result[0])}")

#plotting linear model
plt.plot(df.area, reg.predict(df[['area']]), color = 'blue')
plt.show()

#Generate new CSV file(prediction.csv) with list of home price(areas.csv) predictions
area_df = pd.read_csv("areas.csv")
area_df['prices'] = reg.predict(area_df)
area_df.to_csv("prediction.csv")
