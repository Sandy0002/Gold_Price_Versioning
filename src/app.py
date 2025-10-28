# src/app.py
from fastapi import FastAPI
from src.api import router as main_router
from src.health_checks import router as health_router

app = FastAPI(title="Gold Price Forecasting APIs")

# include all routers
app.include_router(main_router, prefix="/main")
app.include_router(health_router, prefix="/health")

@app.get("/")
def root():
    return {"status": "running"}