import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datetime import datetime

# Load the stock data into a Pandas dataframe
df = pd.read_csv('HDFC.csv')

# Define a threshold for stock price change
threshold = 0.02

# Calculate the percentage change in stock price from the previous day
df['Change'] = df['Close'].pct_change()

# Create a binary classification target variable based on the threshold
df['Target'] = np.where(df['Change'] > threshold, 1, 0)

# Drop any rows with missing values
df.dropna(inplace=True)

# Convert the date column to a datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Split the data into training and testing sets
X_train = df.iloc[:100, 1:-1]
y_train = df.iloc[:100, -1]
X_test = df.iloc[100:, 1:-1]
y_test = df.iloc[100:, -1]

# Create a logistic regression model and fit it to the training data
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Create a new DataFrame with the actual and predicted values and dates
results = pd.DataFrame({'Date': df.iloc[100:, 0], 'Actual': y_test, 'Predicted': y_pred})
results.set_index('Date', inplace=True)

# Plot the actual and predicted target values over time
plt.figure(figsize=(12,6))
plt.plot(results.index, results['Actual'], label='Actual')
plt.plot(results.index, results['Predicted'], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Target')
plt.title('Actual vs. Predicted Target Values')
plt.legend()

# Format the x-axis as dates
plt.gca().xaxis.set_major_formatter(plt.FixedFormatter(df['Date'].dt.strftime('%Y-%m-%d')))
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(len(df.iloc[100:, 0]) // 10))

plt.show()
