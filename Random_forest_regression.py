import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv('AXISBANK.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Preprocess data
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Define features and targets
X = data[['Open', 'High', 'Low', 'Volume']].values
y = data['Returns'].values

# Split data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Perform Random Forest regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# Plot predictions
plt.plot(data.index[train_size:], y_test, label='Actual')
plt.plot(data.index[train_size:], y_pred_rf, label='Random Forest')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Predicted Returns vs Actual Returns')
plt.legend()
plt.show()

print('Random Forest R^2:', r2_rf)
