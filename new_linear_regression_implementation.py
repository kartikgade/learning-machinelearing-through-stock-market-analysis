import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import datetime as dt

# Load the stock data into a Pandas dataframe
df = pd.read_csv('AXISBANK.csv')

# Convert the date column to a numerical format
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = (df['Date'] - dt.datetime(1970,1,1)).dt.total_seconds()

# Split the data into independent and dependent variables
X = df[['Date', 'Close']].values
y = df['Open'].values

# Create a linear regression model and fit it to the data
reg = LinearRegression().fit(X, y)

# Predict the open price using the linear regression model
y_pred = reg.predict(X)

# Create subplots for actual and predicted open prices
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot actual open prices
ax1.plot(df['Date'], y)
ax1.set_ylabel('Actual Open Price')
ax1.set_title('Comparison of Actual and Predicted Open Prices')

# Plot predicted open prices
ax2.plot(df['Date'], y_pred)
ax2.set_xlabel('Date')
ax2.set_ylabel('Predicted Open Price')

plt.show()
