import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the training dataset
train_df = pd.read_csv(r"C:\Users\MBK's EXCALIBUR\Desktop\iyzico-datathon\train.csv")

# Preprocess the data
merchant_id = 'merchant_43992'  # Replace with the merchant_id you want to predict
merchant_data = train_df[train_df['merchant_id'] == merchant_id].copy()

# Feature selection (you may want to include more features based on your analysis)
features = ['month_id', 'net_payment_cnt']
merchant_data = merchant_data[features]

# Convert 'month_id' to datetime and set it as the index
merchant_data['month_id'] = pd.to_datetime(merchant_data['month_id'], format='%Y%m')
merchant_data.set_index('month_id', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(merchant_data)

# Create sequences
def create_sequences(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 1]
        X.append(a)
        Y.append(dataset[i + look_back, 1])
    return np.array(X), np.array(Y)

look_back = 3  # Adjust the look-back period as needed
X, Y = create_sequences(scaled_data, look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(1, look_back)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

# Predict the last quarter (October, November, December) for the specified merchant
last_quarter_months = ['2023-10-01', '2023-11-01', '2023-12-01']
last_quarter_predictions = []

for month in last_quarter_months:
    # Prepare input for prediction
    input_data = np.array(scaled_data[-look_back:, 1]).reshape(1, 1, look_back)
    prediction = model.predict(input_data)
    last_quarter_predictions.append(prediction[0, 0])

# Invert predictions back to original scale
last_quarter_predictions = scaler.inverse_transform(np.array(last_quarter_predictions).reshape(-1, 1))

# Create a DataFrame for the predictions
prediction_df = pd.DataFrame({
    'id': [f'{month.strftime("%Y%m")}merchant_{merchant_id}' for month in pd.to_datetime(last_quarter_months)],
    'net_payment_cnt': last_quarter_predictions.flatten()
})

# Save predictions to CSV
prediction_df.to_csv(r"C:\Users\MBK's EXCALIBUR\Desktop\iyzico-datathon\last_quarter_predictions.csv", index=False)

# Print the predicted values for the last quarter
print(prediction_df)
