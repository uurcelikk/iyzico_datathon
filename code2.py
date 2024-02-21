import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the training dataset
train_df = pd.read_csv(r"C:\Users\MBK's EXCALIBUR\Desktop\iyzico-datathon\train.csv")

# Get unique merchant IDs from the training data
unique_merchant_ids = train_df['merchant_id'].unique()

# Initialize an empty DataFrame to store all predictions
all_predictions_df = pd.DataFrame(columns=['id', 'net_payment_cnt'])

# Loop through each merchant ID
for merchant_id in unique_merchant_ids:
    # Preprocess the data for the current merchant
    merchant_data = train_df[train_df['merchant_id'] == merchant_id].copy()
    features = ['month_id', 'net_payment_cnt']
    merchant_data = merchant_data[features]
    merchant_data['month_id'] = pd.to_datetime(merchant_data['month_id'], format='%Y%m')
    merchant_data.set_index('month_id', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(merchant_data)

    # Create sequences
    look_back = 3  # Adjust the look-back period as needed
    X, Y = create_sequences(scaled_data, look_back)

    # Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # Build and train the LSTM model
    model = build_and_train_lstm_model(X, Y, epochs=100)

    # Predict the last quarter (October, November, December) for the current merchant
    last_quarter_months = ['2023-10-01', '2023-11-01', '2023-12-01']
    last_quarter_predictions = predict_last_quarter(model, scaled_data, look_back, last_quarter_months, scaler)

    # Create a DataFrame for the predictions
    prediction_df = pd.DataFrame({
        'id': [f'{month.strftime("%Y%m")}_{merchant_id}' for month in pd.to_datetime(last_quarter_months)],
        'net_payment_cnt': last_quarter_predictions.flatten()
    })

    # Append the predictions to the overall DataFrame
    all_predictions_df = all_predictions_df.append(prediction_df, ignore_index=True)

# Save all predictions to CSV
all_predictions_df.to_csv(r"C:\Users\MBK's EXCALIBUR\Desktop\iyzico-datathon\all_merchant_predictions.csv", index=False)

# Print the overall predictions DataFrame
print(all_predictions_df)
