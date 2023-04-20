import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#import pandas as pd
import datetime as dt
#from sklearn.linear_model import LinearRegression

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

# Plot the actual and predicted open prices over time
plt.plot(df['Date'], y, label='Actual Open Price')
plt.plot(df['Date'], y_pred, label='Predicted Open Price')

plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Stock Data Analysis using Linear Regression')
plt.legend()
plt.show()
