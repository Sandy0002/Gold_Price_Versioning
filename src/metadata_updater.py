import json
from pathlib import Path
import datetime

# Paths
project_root = Path(__file__).resolve().parents[1]
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)
model_file = models_dir / "gold_lstm_model.h5"
metadata_file = models_dir / "model_metadata.json"

# Example new model info
new_model_info = {
    "name": "gold_lstm_model",
    "trained_at": str(datetime.datetime.now()),
    "r2_score": 0.98
}

# Check if metadata file exists
if metadata_file.exists():
    # Load existing metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    # Compare R² score
    old_r2 = metadata.get("r2_score", -float("inf"))
    if new_model_info["r2_score"] > old_r2:
        print(f"✅ New model R² ({new_model_info['r2_score']}) is better than old ({old_r2})")
        # Update metadata
        metadata.update(new_model_info)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

    else:
        print(f"⚠️ New model R² ({new_model_info['r2_score']}) is worse. Keeping old model")
else:
    # No metadata exists → save new
    print("ℹ️ No existing metadata. Saving new model and metadata.")
    with open(metadata_file, "w") as f:
        json.dump(new_model_info, f, indent=4)
