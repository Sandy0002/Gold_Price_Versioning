from fastapi import FastAPI
from src.api import app as main_api
from src.health_checks import app as health_api

# ----------------------------
# Master FastAPI application
# ----------------------------
app = FastAPI(title="Gold Price Forecasting - Unified API", version="1.0")

# Mount sub-apps under clean URL prefixes
app.mount("/", main_api)
app.mount("/health", health_api)

# Root route for sanity check
@app.get("/status")
def status():
    return {"status": "running", "message": "Unified API is live on Render"}

# ----------------------------
# Optional: run locally
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
