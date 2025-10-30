# In this program we will be training data by taking inputs from sequence_creator.py 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from src.sequence_creator import prepare_data
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
from src_logger.logger import get_logger


logger = get_logger(__name__)

def modelUpdater(newModel,xTest,yTest):
    logger.info("Updating model")

    try:
        project_root = Path(__file__).resolve().parents[1]

        # Create models directory under project root
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "gold_lstm_model.h5"

        # Evaluate new model
        y_pred_new = newModel.predict(xTest)
        new_score_mse = mean_squared_error(yTest, y_pred_new)
        new_score_r2 = r2_score(yTest, y_pred_new)

        # Check if existing model exists
        if model_path.exists():
            # Load and evaluate old model
            old_model = load_model(model_path)
            y_pred_old = old_model.predict(xTest)
            logger.debug("ℹ️Generated predictions using old model")
            old_score_mse = mean_squared_error(yTest, y_pred_old)
            old_score_r2 = r2_score(yTest, y_pred_old)

            logger.debug(f"Old model MSE: {old_score_mse:.5f}")
            logger.debug(f"New model MSE: {new_score_mse:.5f}")

            # Compare and decide
            if new_score_r2 > old_score_r2:  # Lower MSE = better model
                logger.info("New model is better. Saving it.")
                newModel.save(model_path, save_format="h5")

            else:
                logger.info("New model is worse. Keeping old one.")
        else:
            # No existing model → save the first one
            logger.info("No existing model found. Saving new model.")
            newModel.save(model_path)
    except Exception as e:
        logger.exception("Error occured while updating model")
        raise


def training():
    # Getting data from sequence_creator
    logger.info("Training the model")
    try:
        logger.debug("Preparing data")
        xTrain,xTest,yTrain,yTest,scaler = prepare_data()
        logger.debug("Prepared data")

        model = Sequential([
            # IN layer 1: 50 : Number of neurons, input_shape: timesteps,features, dropout is to remove some neurons randomly to avoid overfitting, return_sequence is for  outputs the entire sequence to the next LSTM layer (needed when stacking LSTMs).
            LSTM(50, return_sequences=True, input_shape=(xTrain.shape[1], 1)), Dropout(0.2),
            LSTM(50, return_sequences=False), Dropout(0.2),

            # Fully connected layer with 25 neurons. ReLU helps capture non-linear relationships between past and future prices.
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Fitting data
        history = model.fit(xTrain, yTrain, epochs=20, batch_size=32, validation_data=(xTest, yTest), verbose=1)
        
        logger.debug("Trained new model")

        newModel = model
        # Get project root (assuming current file is inside src/)
        modelUpdater(newModel,xTest,yTest)
        logger.info("Updated model and sending xTest,yTest and scaler")

        return xTest,yTest,scaler

    except Exception as e:
        logger.exception("Error occured during model training")
        raise