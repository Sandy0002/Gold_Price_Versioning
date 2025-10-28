from fastapi import FastAPI
from src.api import router as api_router
from src.health_checks import router as health_router

app = FastAPI(title="Gold Price Forecasting - Unified API", version="1.0")

app.include_router(api_router)
app.include_router(health_router)               # Health check routes (/health/…)


@app.get("/status")
def status():
    return {"status": "running"}


# Optional: run locally
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
