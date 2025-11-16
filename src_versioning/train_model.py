# In this program we will be training data by taking inputs from a preprocessed CSV
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import argparse
import pandas as pd
import sys
import joblib
import boto3
from src_versioning.sequence_creator import prepare_data


# ------------------ DATA LOADING ------------------ #
def load_data(input_path):
    try:
        df = pd.read_csv(input_path)
        print(f"📥 Loaded data from {input_path} — shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()  # return empty df if file missing


# ------------------ MODEL UPDATER ------------------ #
def modelUpdater(newModel, xTest, yTest):
    # Local storage
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "gold_lstm_model.pkl"

    # S3 setup
    bucket = "mlops-model-store01"
    s3_key = "models/gold_lstm_model.pkl"
    local_path = "models/gold_lstm_model.pkl"

    s3 = boto3.client("s3")

    # Download from S3
    try:
        s3.download_file(bucket, s3_key, model_path)
    except Exception as e:
        pass

    # ------ NEW MODEL METRICS ------
    y_pred_new = newModel.predict(xTest)
    new_mse = mean_squared_error(yTest, y_pred_new)
    new_r2 = r2_score(yTest, y_pred_new)

    print(f"New Model ➝ MSE: {new_mse:.5f}, R2: {new_r2:.5f}")

    # ------ If local model exists → compare ------
    if model_path.exists():
        print("ℹ️ Old model found. Comparing performance...")

        old_model = joblib.load(model_path)
        y_pred_old = old_model.predict(xTest)
        old_mse = mean_squared_error(yTest, y_pred_old)
        old_r2 = r2_score(yTest, y_pred_old)

        print(f"Old Model ➝ MSE: {old_mse:.5f}, R2: {old_r2:.5f}")

        # ------ METRIC COMPARISON ------
        if new_r2 > old_r2:
            print("✅ New model is better → Saving locally + Uploading to S3")

            joblib.dump(newModel, model_path)

            # Upload to S3
            s3.upload_file(str(model_path), bucket, s3_key)
        else:
            print("⚠️ New model worse → Keeping old model")

    else:
        # ------ FIRST TIME TRAINING ------
        print("ℹ️ No old model → Saving first model locally + uploading to S3")
        
        joblib.dump(newModel, model_path)
        s3.upload_file(str(model_path), bucket, s3_key)

    print("✔️ Update step completed.")

# ------------------ TRAINING ------------------ #
def training():
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    if X_train is None or len(X_train) == 0:
        print("⚠️ No training data available, skipping model training.")
        sys.exit(0)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    modelUpdater(model, X_test, y_test)
    return X_test,y_test,scaler


# ------------------ MAIN ------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = load_data(args.input)
    trained_model = training()

    # The model is already saved inside modelUpdater, but save output path too for DVC tracking
    # trained_model.save(args.output)
    # joblib.dump(trained_model, args.output)
    print(f"✅ Model saved in s3")