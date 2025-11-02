# In this program we will be training data by taking inputs from a preprocessed CSV

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import sys


# ------------------ DATA LOADING ------------------ #
def load_data(input_path):
    try:
        df = pd.read_csv(input_path)
        print(f"📥 Loaded data from {input_path} — shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()  # return empty df if file missing


# ------------------ SEQUENCE CREATION ------------------ #
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ------------------ MODEL UPDATER ------------------ #
def modelUpdater(newModel, xTest, yTest):
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "gold_lstm_model.keras"

    y_pred_new = newModel.predict(xTest)
    new_score_mse = mean_squared_error(yTest, y_pred_new)
    new_score_r2 = r2_score(yTest, y_pred_new)

    if model_path.exists():
        old_model = load_model(model_path)
        y_pred_old = old_model.predict(xTest)
        print("ℹ️ Generated predictions using old model")
        old_score_mse = mean_squared_error(yTest, y_pred_old)
        old_score_r2 = r2_score(yTest, y_pred_old)

        print(f"Old model MSE: {old_score_mse:.5f}, R2: {old_score_r2:.5f}")
        print(f"New model MSE: {new_score_mse:.5f}, R2: {new_score_r2:.5f}")

        if new_score_r2 > old_score_r2:
            print("✅ New model is better. Saving it.")
            newModel.save(model_path)
        else:
            print("⚠️ New model is worse. Keeping old one.")
    else:
        print("ℹ️ No existing model found. Saving new model.")
        newModel.save(model_path)


# ------------------ DATA PREPARATION ------------------ #
def prepare_data(df):
    if df is None or df.empty:
        print("⚠️ Empty dataframe received — skipping training.")
        return None, None, None, None, None

    data = df[["Close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    lookback = 60

    X, y = create_sequences(scaled_data, lookback)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler


# ------------------ TRAINING ------------------ #
def training(data):
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)

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
    return model


# ------------------ MAIN ------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = load_data(args.input)
    trained_model = training(df)

    # The model is already saved inside modelUpdater, but save output path too for DVC tracking
    trained_model.save(args.output)
    print(f"✅ Model saved to {args.output}")