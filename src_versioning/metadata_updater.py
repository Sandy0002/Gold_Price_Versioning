import json
from pathlib import Path
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Paths
project_root = Path(__file__).resolve().parents[1]
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)
# model_file = models_dir / "gold_lstm_model.h5"
model_file = models_dir / "gold_lstm_model.keras"
metadata_file = models_dir / "model_metadata.json"

# Example new model info
new_model_info = {
    "name": "gold_lstm_model",
    "trained_at": str(datetime.datetime.now()),
}

# Check if metadata file exists
if metadata_file.exists():
    # Load existing metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        # Update metadata
        metadata.update(new_model_info)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

else:
    # No metadata exists → save new
    print("ℹ️ No existing metadata. Saving new model and metadata.")
    with open(metadata_file, "w") as f:
        json.dump(new_model_info, f, indent=4)
