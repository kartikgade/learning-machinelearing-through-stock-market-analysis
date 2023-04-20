import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv('HDFC.csv')
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

# Perform Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Perform SVM
svm = SVR(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
r2_svm = r2_score(y_test, y_pred_svm)

# Plot predictions
plt.plot(data.index[train_size:], y_test, label='Actual')
plt.plot(data.index[train_size:], y_pred_ridge, label='Ridge Regression')
plt.plot(data.index[train_size:], y_pred_svm, label='SVM')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Predicted Returns vs Actual Returns')
plt.legend()
plt.show()

print('Ridge Regression R^2:', r2_ridge)
print('SVM R^2:', r2_svm)
