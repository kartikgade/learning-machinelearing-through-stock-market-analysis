import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from datetime import datetime

# Load the stock data into a Pandas dataframe
df = pd.read_csv('HDFC.csv')

# Convert the date column to a datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Create a new dataframe with only the Date and Open columns
df_open = df[['Date', 'Open']]

# Split the data into training and testing sets
train = df_open.iloc[:100,:]
test = df_open.iloc[100:,:]

# Extract the training and testing features and targets
X_train = train['Date'].apply(lambda x: x.toordinal()).values.reshape(-1, 1)
y_train = train['Open'].values.reshape(-1, 1)
X_test = test['Date'].apply(lambda x: x.toordinal()).values.reshape(-1, 1)
y_test = test['Open'].values.reshape(-1, 1)

# Create polynomial features for the training and testing features
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test_poly)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

# Print the R-squared value
print("R-squared: {:.2f}".format(r2))

# Plot the actual and predicted stock prices over time
plt.figure(figsize=(12,6))
plt.plot(test['Date'], y_test, label='Actual')
plt.plot(test['Date'], y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()

# Format the x-axis as dates
plt.gca().xaxis.set_major_formatter(plt.FixedFormatter(df['Date'].dt.strftime('%Y-%m-%d')))
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(len(df.iloc[100:, 0]) // 10))

plt.show()
