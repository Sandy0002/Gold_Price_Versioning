FROM python:3.11-slim

# ---------- Environment Variables ----------
ENV PYTHONUNBUFFERED=1 \
    APP_ENV=docker \
    LOG_LEVEL=INFO \
    PORT=8000

# ---------- Set Working Directory ----------
WORKDIR /app

# Add /app to Python path so imports work
ENV PYTHONPATH="${PYTHONPATH}:/app"
# ---------- Copy Dependency File and Install ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---------- Copy Only Necessary Folders ----------
COPY ./models ./models
# COPY ./src_logger ./src_logger
COPY ./src_versioning ./src_versioning
COPY .dvc, dvc.yaml, dvc.lock

# ---------- (Optional) Copy .env if needed ----------
# COPY .env ./

# ---------- Create Logs Directory ----------
RUN mkdir -p /app/logs  && chmod -R 777 /app/logs

# ---------- Expose Port ----------
EXPOSE 8000

# ---------- Run Application ----------
# CMD ["uvicorn", "src_versioning.api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD dvc pull && uvicorn src_versioning.api:app --host 0.0.0.0 --port $PORT