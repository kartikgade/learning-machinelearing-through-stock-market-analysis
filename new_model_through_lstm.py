import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the stock data into a Pandas dataframe
df = pd.read_csv('AXISBANK.csv')

# Convert the date column to a numerical format
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = (df['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Open'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_data = scaled_data[:int(0.8*len(df))]
test_data = scaled_data[int(0.8*len(df)):]

# Convert the data into sequences for LSTM
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 60

X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape the data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Make predictions on the test data
predictions = model.predict(X_test)

# Scale the predictions back to their original range
predictions = scaler.inverse_transform(predictions)

# Scale the actual values back to their original range
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the actual and predicted values
plt.plot(y_test, label='Actual Open Price')
plt.plot(predictions, label='Predicted Open Price')
plt.xlabel('Time')
plt.ylabel('Open Price')
plt.title('Stock Data Analysis using LSTM')
plt.legend()
plt.show()

