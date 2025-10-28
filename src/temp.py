from keras.models import load_model
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
models_dir =project_root / "models"
model_path = models_dir /"gold_lstm_model.h5"

# Load your old model (.h5 format)
model = load_model(model_path, safe_mode=False)

# Save it in the new format inside 'models' folder
model.save(models_dir/"gold_lstm_model.keras")

print("✅ Model saved successfully in models/gold_lstm_model.keras")