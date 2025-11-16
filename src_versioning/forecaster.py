# This program will generate predictions for future period
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src_versioning.train_model import training
import joblib
import boto3
from sklearn.preprocessing import MinMaxScaler

xTest,yTest,scaler = training()

# # Importing model
# bucket = "mlops-model-store01"
# s3_key = "models/gold_lstm_model.pkl"

# local_path = "gold_lstm_model.pkl"

s3 = boto3.client("s3")

# Download from S3
# s3.download_file(bucket, s3_key, local_path)
model_path = "models/gold_lstm_model.pkl"

# Load model
model = joblib.load(model_path)

predictions = model.predict(xTest)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual = scaler.inverse_transform(yTest.reshape(-1, 1))

mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, predictions)
r2 = r2_score(actual, predictions)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root MSE (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")