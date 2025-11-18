FROM python:3.11-slim

# Environment vars
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC with S3 support
RUN pip install --no-cache-dir "dvc[s3]"

# Copy entire project (safer than selecting subfolders)
COPY . .

# Create logs directory
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

# Expose port
EXPOSE 8000

# Run: pull model from DVC -> start FastAPI
CMD dvc pull && uvicorn src_logger.api:app --host 0.0.0.0 --port $PORT