# IN this program we will fetch data from data folder and create sequence of samples which will be input for model training

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from src.data_preprocess import fetch_data_postgres
import os

# Create sequences (lookback = 60 days)
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def prepare_data():
    df = fetch_data_postgres()
    data = df[["Close"]]

    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print(scaled_data)
    lookback = 60

    X, y = create_sequences(scaled_data, lookback)

    # That line reshapes X so it matches the 3D input format LSTMs expect: [samples, timesteps, features]
    '''
    X.shape[0] → number of samples  
    X.shape[1] → number of timesteps (lookback window, e.g. 60)  
    1 → number of features per timestep (e.g. just "Close" price)

    So if your original X was shaped like (2000, 60) — meaning 2000 sequences, each with 60 values — the reshape makes it (2000, 60, 1) so the LSTM knows there’s 1 feature per time step.
'''
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]

    # Train / test split
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train,X_test,y_train,y_test,scaler