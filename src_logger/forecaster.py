# This program will generate predictions for future period
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.train_model import training
from src_logger.logger import get_logger


logger = get_logger(__name__)


try: 
      xTest,yTest,scaler = training()

      # Importing model
      project_root = Path(__file__).resolve().parents[1]
      models_dir = project_root / "models"
      # model_path = models_dir / "gold_lstm_model.h5"
      model_path = models_dir / "gold_lstm_model.keras"
      model = load_model(model_path)

      predictions = model.predict(xTest)
      predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
      actual = scaler.inverse_transform(yTest.reshape(-1, 1))

      mse = mean_squared_error(actual, predictions)
      rmse = np.sqrt(mse)
      mae = mean_absolute_error(actual, predictions)
      r2 = r2_score(actual, predictions)

      logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
      logger.info(f"Root MSE (RMSE): {rmse:.2f}")
      logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
      logger.info(f"R² Score: {r2:.2f}")

except Exception as e:
        logger.exception("Error occurred getting model metrics.")
        raise